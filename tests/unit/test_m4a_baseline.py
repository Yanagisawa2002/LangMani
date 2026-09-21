"""Fast M4A contracts. All data and simulator objects here are explicitly synthetic."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from langmani.datasets.lerobot_types import FeatureContract, LeRobotExportManifest
from langmani.datasets.splits import build_split_assignments
from langmani.environments.specs import TaskSpec, canonical_instruction, stable_task_id
from langmani.policies.m4a_data import (
    ACTION,
    CANONICAL_TEXTS,
    IMAGE,
    STATE,
    TASK,
    TASK_ID,
    Moments,
    load_manifest,
    local_dataset_only,
    make_split,
    validate_rows,
    write_json,
)
from langmani.policies.m4a_evaluation import aggregate, evaluation_seeds, run_act, state_digest
from langmani.policies.m4a_training import (
    act_config,
    checkpoint_directory,
    policy_observation,
    train_statistics,
    validate_act_config,
)


class DatasetFixture:
    def __init__(self) -> None:
        texts = sorted(CANONICAL_TEXTS)
        self.features = FeatureContract().to_lerobot_features() | {
            key: {"dtype": "int64", "shape": (1,), "names": None}
            for key in ("index", "frame_index", "episode_index", "task_index", "timestamp")
        }
        self.fps, self.num_frames, self.num_episodes = 20, 18, 6
        self.meta = SimpleNamespace(
            info=SimpleNamespace(codebase_version="v3.0"),
            total_frames=18,
            total_episodes=6,
            camera_keys=[IMAGE],
            tasks=pd.DataFrame({"task_index": range(6)}, index=texts),
            episodes=[
                {
                    "length": 3,
                    "dataset_from_index": i * 3,
                    "dataset_to_index": i * 3 + 3,
                    "tasks": [text],
                }
                for i, text in enumerate(texts)
            ],
        )
        self.rows = [
            {
                IMAGE: np.zeros((3, 256, 256), np.uint8),
                STATE: np.full(9, i, np.float32),
                ACTION: np.full(8, i + j, np.float32),
                "task": text,
                "task_index": np.int64(i),
                "index": np.int64(i * 3 + j),
                "episode_index": np.int64(i),
                "frame_index": np.int64(j),
                "timestamp": np.float32(j / 20),
            }
            for i, text in enumerate(texts)
            for j in range(3)
        ]
        self.hf_dataset = self.rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def test_validator_exhaustively_validates_counts_stats_and_last_frame() -> None:
    dataset = DatasetFixture()
    report = validate_rows(dataset)
    assert report["num_frames"] == 18 and report["num_episodes"] == 6
    assert report["statistics"][STATE]["mean"] == [2.5] * 9
    dataset.rows[-1][ACTION][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_rows(dataset)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "privileged",
        "camera",
        "image_shape",
        "image_dtype",
        "state",
        "action",
        "inf",
        "frame",
        "timestamp",
        "index",
        "episode",
        "boundary",
        "length",
        "fps",
        "task",
        "sentences",
        "count",
    ],
)
def test_validator_rejects_malformed_data(case: str) -> None:
    d = DatasetFixture()
    row = d.rows[1]
    if case == "missing":
        del row[STATE]
    elif case == "privileged":
        d.features["observation.cube_poses"] = {}
    elif case == "camera":
        d.meta.camera_keys.append("wrist_camera")
    elif case == "image_shape":
        row[IMAGE] = np.zeros((256, 256, 3), np.uint8)
    elif case == "image_dtype":
        row[IMAGE] = row[IMAGE].astype(np.float32)
    elif case == "state":
        row[STATE] = np.zeros(10, np.float32)
    elif case == "action":
        row[ACTION] = np.zeros(9, np.float32)
    elif case == "inf":
        row[STATE][0] = np.inf
    elif case == "frame":
        row["frame_index"] = np.int64(0)
    elif case == "timestamp":
        row["timestamp"] = np.float32(0.15)
    elif case == "index":
        row["index"] = np.int64(3)
    elif case == "episode":
        row["episode_index"] = np.int64(3)
    elif case == "boundary":
        d.meta.episodes[-1]["dataset_to_index"] += 1
    elif case == "length":
        d.meta.episodes[0]["length"] += 1
    elif case == "fps":
        d.fps = 30
    elif case == "task":
        row["task"] = "a new sentence"
    elif case == "sentences":
        d.meta.tasks.index = list(d.meta.tasks.index)[:-1] + ["extra"]
    elif case == "count":
        d.meta.total_frames = 19
    with pytest.raises((ValueError, KeyError, IndexError)):
        validate_rows(d)


def full_manifest() -> LeRobotExportManifest:
    # Reuse existing typed M3B construction helpers; no fake source is accepted by production CLI.
    from test_lerobot_contract import _config, _episode_record

    from langmani.datasets.identity import COLLECTION_SCHEMA_VERSION
    from langmani.datasets.lerobot_types import (
        LEROBOT_EXPORT_SCHEMA_VERSION,
        stable_export_fingerprint,
    )

    config = _config()
    groups = [f"scene-{i}" for i in range(60)]
    assignments = build_split_assignments(groups, config.split_config)
    by_group = {a.scene_group_id: a.split for a in assignments}
    records = []
    for index in range(360):
        group = groups[index // 6]
        record = _episode_record(index, group_id=group, split=by_group[group])
        task = TaskSpec(record.target_object_id, record.target_bin_id, "canonical_v0")
        records.append(
            replace(
                record,
                task_id=stable_task_id(task),
                canonical_instruction=canonical_instruction(task),
                source_scene_seed=index // 6,
            )
        )
    ids = tuple(r.source_episode_id for r in records)
    return LeRobotExportManifest(
        export_schema_version=LEROBOT_EXPORT_SCHEMA_VERSION,
        export_fingerprint=stable_export_fingerprint(config, ids),
        source_collection_run_id=config.expected_source_collection_run_id,
        source_run_fingerprint=config.expected_source_run_fingerprint,
        source_archive_digest=config.expected_source_archive_digest,
        source_schema_version=COLLECTION_SCHEMA_VERSION,
        repo_id=config.repo_id,
        config=config,
        ordered_source_episode_ids=ids,
        split_assignments=assignments,
        episodes=tuple(records),
        runtime_versions={"fixture": "true"},
        finalized=True,
    )


def test_split_reproducible_inherits_scene_seed_and_never_uses_test() -> None:
    manifest = full_manifest()
    validation = {
        "content_sha256": "fixture",
        "num_frames": sum(r.output_frame_count for r in manifest.episodes),
    }
    split = make_split(manifest, validation)
    assert split == make_split(manifest, validation)
    assert split["seed"] == 17 and split["task_name"] == TASK_ID
    assert len(split["train_episode_ids"]) == 48
    assert len(split["held_out_episode_ids"]) == len(split["excluded_test_episode_ids"]) == 6
    assert not set(split["train_episode_ids"]) & set(split["held_out_episode_ids"])
    assert all(i % 6 == 0 for i in split["train_episode_ids"])
    assert split["total_frames"] == sum(
        split[f"{name}_frames"] for name in ("train", "held_out", "excluded_test")
    )


def test_schema_no_privileged_information_or_language_and_no_whole_dataset_stats() -> None:
    config = act_config("cpu")
    validate_act_config(config)
    config.input_features["observation.environment_state"] = config.input_features[STATE]
    with pytest.raises(ValueError, match="allowlist"):
        validate_act_config(config)
    batch = policy_observation(np.zeros((3, 256, 256), np.uint8), np.zeros(9, np.float32))
    assert set(batch) == {IMAGE, STATE}
    d = DatasetFixture()
    with pytest.raises(ValueError, match="held-out"):
        train_statistics(d, [0])
    d.rows = d.rows[:3]
    d.hf_dataset = d.rows
    d.meta.stats = {"DO_NOT_USE": 1e9}
    stats = train_statistics(d, [0])
    assert stats[STATE]["mean"] == [0] * 9 and stats[ACTION]["mean"] == [1] * 8


def test_metrics_population_standard_deviation_and_seed_schedule() -> None:
    summary = aggregate(
        [
            {
                "success": True,
                "length": 2,
                "termination_reason": "success",
                "target_off_table": False,
            },
            {
                "success": False,
                "length": 6,
                "termination_reason": "timeout",
                "target_off_table": False,
            },
        ]
    )
    assert summary["success_rate"] == 0.5
    assert summary["mean_episode_length"] == 4 and summary["std_episode_length"] == 2
    assert summary["termination_reason_counts"] == {"success": 1, "timeout": 1}
    with pytest.raises(ValueError):
        aggregate([])
    split = {"all_source_scene_seeds": list(range(60))}
    assert evaluation_seeds(split, 20, 0) == evaluation_seeds(split, 20, 0)
    assert len(set(evaluation_seeds(split, 20, 0))) == 20
    assert not set(evaluation_seeds(split, 20, 0)) & set(range(60))
    assert state_digest({"x": np.zeros(3)}) != state_digest({"x": np.ones(3)})


def test_checkpoint_and_output_paths(tmp_path: Path) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        checkpoint_directory(tmp_path / "missing")
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint_directory(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "act_baseline_cli", Path(__file__).parents[2] / "scripts/act_baseline.py"
    )
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(ValueError, match="overlap"):
        cli.safe_output(tmp_path / "dataset/result", tmp_path / "dataset")
    args = cli.parse_args(
        [
            "train",
            "--dataset-root",
            "data with spaces",
            "--split",
            "split.json",
            "--output",
            "output with spaces",
            "--mode",
            "smoke",
        ]
    )
    assert str(args.dataset_root) == "data with spaces"


def test_rollout_observation_allowlist_resets_queue_and_rejects_invalid_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langmani.datasets.observation_reconstruction as reconstruction
    import langmani.datasets.policy_state as state_module

    monkeypatch.setattr(
        reconstruction, "extract_base_camera_rgb", lambda obs: np.zeros((256, 256, 3), np.uint8)
    )
    monkeypatch.setattr(
        state_module, "extract_panda_policy_state_v0", lambda robot: np.zeros(9, np.float32)
    )

    class Policy:
        resets = 0

        def reset(self) -> None:
            self.resets += 1

        def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
            assert set(batch) == {IMAGE, STATE}
            return torch.full((1, 8), 2.0)

    class Processor:
        def reset(self) -> None:
            pass

        def __call__(self, value: Any) -> Any:
            return value

    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            single_action_space=SimpleNamespace(shape=(8,), low=np.full(8, -1), high=np.ones(8)),
            agent=SimpleNamespace(robot=object()),
        )
    )
    policy = Policy()
    result = run_act(
        env,
        {"privileged": "never forwarded", "task": canonical_instruction(TASK)},
        policy,
        Processor(),
        Processor(),
    )
    assert policy.resets == 1
    assert result["length"] == 0 and result["termination_reason"] == "action_out_of_bounds"


@pytest.mark.parametrize("gripper", [1.01, -1.2, 0.4, float("nan"), float("inf")])
def test_rollout_native_gripper_saturation_preserves_arm_and_audits(
    monkeypatch: pytest.MonkeyPatch, gripper: float
) -> None:
    import langmani.datasets.observation_reconstruction as reconstruction
    import langmani.datasets.policy_state as state_module

    monkeypatch.setattr(
        reconstruction, "extract_base_camera_rgb", lambda obs: np.zeros((256, 256, 3), np.uint8)
    )
    monkeypatch.setattr(
        state_module, "extract_panda_policy_state_v0", lambda robot: np.zeros(9, np.float32)
    )
    prediction = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, gripper]])
    original = prediction.clone()
    actions: list[np.ndarray] = []

    class Policy:
        def reset(self) -> None:
            pass

        def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
            assert set(batch) == {IMAGE, STATE}
            return prediction

    class Processor:
        def reset(self) -> None:
            pass

        def __call__(self, value: Any) -> Any:
            return value

    def step(action: np.ndarray) -> tuple[Any, ...]:
        actions.append(action.copy())
        return (
            {},
            0.0,
            np.array(False),
            np.array(len(actions) == 2),
            {
                "success": np.array(False),
                "target_off_table": np.array(False),
            },
        )

    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            single_action_space=SimpleNamespace(shape=(8,), low=np.full(8, -1), high=np.ones(8)),
            agent=SimpleNamespace(robot=object()),
        ),
        step=step,
    )
    result = run_act(env, {}, Policy(), Processor(), Processor(), max_steps=2)
    np.testing.assert_equal(prediction.numpy(), original.numpy())
    if not np.isfinite(gripper):
        assert not actions
        assert result["termination_reason"] == "invalid_action" and result["length"] == 0
        assert result["gripper_saturation_count"] == 0
        return
    assert result["length"] == 2 and result["termination_reason"] == "timeout"
    for action in actions:
        assert action.dtype == np.float32
        np.testing.assert_array_equal(action[:7], original.numpy()[0, :7])
        assert action[7] == pytest.approx(max(-1.0, min(1.0, gripper)))
    assert result["gripper_saturation_count"] == (2 if abs(gripper) > 1 else 0)
    assert result["gripper_max_overshoot"] == pytest.approx(max(0.0, abs(gripper) - 1))
    summary = aggregate([result])
    assert summary["gripper_saturation_count"] == result["gripper_saturation_count"]
    assert summary["gripper_max_overshoot"] == result["gripper_max_overshoot"]


def test_moments_matches_numpy() -> None:
    x = np.random.default_rng(0).normal(size=(23, 9))
    moments = Moments(9)
    for chunk in (x[:2], x[2:20], x[20:]):
        moments.update(chunk)
    np.testing.assert_allclose(moments.report()["mean"], x.mean(0))
    np.testing.assert_allclose(moments.report()["std"], x.std(0))


@pytest.mark.parametrize("failure", [None, "exit", "seed"])
def test_isolated_expert_uses_selected_interpreter_and_checks_worker_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    import langmani.experts.runtime as runtime
    import langmani.policies.m4a_evaluation as evaluation

    interpreter = str(tmp_path / "planner with spaces" / "python")
    monkeypatch.setattr(runtime, "resolve_planner_python", lambda: interpreter)

    def run(command: list[str], **kwargs: Any) -> Any:
        assert command[0] == interpreter
        assert command[1:3] == ["-m", "langmani.policies.m4a_evaluation"]
        assert kwargs["timeout"] == 600 and kwargs["check"] is False
        write_json(
            Path(command[-1]),
            {
                "seed": 8 if failure == "seed" else 7,
                "sim_backend": "physx_cpu",
                "success": True,
                "initial_state_sha256": "a" * 64,
            },
        )
        return SimpleNamespace(returncode=1 if failure == "exit" else 0, stdout="worker", stderr="")

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    if failure:
        with pytest.raises((RuntimeError, ValueError), match="exited|identity"):
            evaluation.isolated_expert(7, "physx_cpu", tmp_path)
    else:
        result = evaluation.isolated_expert(7, "physx_cpu", tmp_path)
        assert result["success"] is True and "seed" not in result
    assert (tmp_path / "expert_workers/seed-7.log").read_text() == "worker"


def test_manifest_gate_rejects_task_or_scene_seed_leakage(tmp_path: Path) -> None:
    manifest = full_manifest()
    sidecar = tmp_path / "langmani"
    write_json(sidecar / "export_manifest.json", manifest.to_dict())
    write_json(
        sidecar / "complete.json",
        {
            "schema_version": "langmani-m3b-completion-v1",
            "export_fingerprint": manifest.export_fingerprint,
        },
    )
    write_json(
        sidecar / "validation_report.json",
        {
            "passed": True,
            "export_fingerprint": manifest.export_fingerprint,
            **{
                key: True
                for key in (
                    "source_alignment_validated",
                    "video_decode_validated",
                    "feature_schema_validated",
                    "privileged_leakage_validated",
                )
            },
        },
    )
    assert load_manifest(tmp_path).to_dict() == manifest.to_dict()
    changed = manifest.to_dict()
    changed["episodes"][6]["source_scene_seed"] = 0
    write_json(sidecar / "export_manifest.json", changed)
    with pytest.raises(ValueError, match="seed"):
        load_manifest(tmp_path)
    changed = manifest.to_dict()
    changed["episodes"][0]["task_id"] = "wrong-task"
    write_json(sidecar / "export_manifest.json", changed)
    with pytest.raises(ValueError, match="TaskSpec"):
        load_manifest(tmp_path)


def test_local_dataset_guard_forbids_automatic_repair() -> None:
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    with local_dataset_only():
        with pytest.raises(ValueError, match="repair"):
            LeRobotDataset._download(None)
        with pytest.raises(ValueError, match="repair"):
            LeRobotDatasetMetadata._pull_from_repo(None)


@pytest.mark.parametrize("mismatch,act_steps", [(False, 10), (True, 10), (False, 0)])
def test_paired_evaluation_artifacts_and_invalid_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: bool,
    act_steps: int,
) -> None:
    import langmani.experts as experts
    import langmani.policies.m4a_evaluation as evaluation

    sequence: list[int] = []

    class Environment:
        def __init__(self) -> None:
            self.unwrapped = self
            self.offset = len(sequence) % 2 if mismatch else 0

        def reset(
            self, *, seed: int, options: dict[str, Any]
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            assert options == {"task_spec": TASK.to_dict()}
            self.seed = seed
            sequence.append(seed)
            return {}, {}

        def get_state_dict(self) -> dict[str, Any]:
            return {"state": np.array([self.seed, self.offset])}

        def close(self) -> None:
            pass

    class Expert:
        def __init__(self, env: Any, config: Any) -> None:
            assert config.max_episode_steps == 200

        def run(self) -> Any:
            return SimpleNamespace(
                exception_type=None,
                status=SimpleNamespace(value="success"),
                total_environment_steps=5,
                final_environment_evaluation={"success": True},
            )

    monkeypatch.setattr(experts, "PickPlaceExpert", Expert)
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *args: (None, None, None))
    monkeypatch.setattr(
        evaluation,
        "run_act",
        lambda *args: {
            "success": False,
            "length": act_steps,
            "termination_reason": "timeout",
            "target_off_table": False,
        },
    )
    result = evaluation.evaluate(
        checkpoint=tmp_path / "checkpoint",
        output=tmp_path / "eval",
        split={"all_source_scene_seeds": [0]},
        count=2,
        seed=4,
        device="cpu",
        sim_backend="physx_cpu",
        provenance={"fixture": True},
        environment_factory=Environment,
    )
    assert sequence[0] == sequence[1] and sequence[2] == sequence[3]
    valid = not mismatch and act_steps > 0
    assert result["passed"] is valid
    assert result["closed_loop_evaluated"] is False
    assert result["absolute_success_rate_gap"] == (1.0 if valid else None)
    assert (tmp_path / "eval/metrics.json").is_file()
    assert len((tmp_path / "eval/episodes.csv").read_text().splitlines()) == 5
