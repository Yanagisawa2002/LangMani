"""Fixture-only tests for the controlled language intervention and its provenance."""

from __future__ import annotations

import copy
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from langmani.policies.m4a_data import digest, file_digest, write_json
from langmani.policies.m4b_protocol import GOALS
from langmani.policies.m4c_evaluation import diagnostics, metrics, scored_rows, validate_rollout
from langmani.policies.m4c_language import (
    catalogs,
    check_manifest,
    make_manifest,
    make_schedule,
    normalize,
    sample_expression,
    validate_catalogs,
)
from langmani.policies.m4c_training import (
    LanguageDataset,
    LanguageSamples,
    consolidate_samples,
    model_state_hash,
)


def split() -> dict:
    train, _ = catalogs()
    return {
        "train_episode_ids": [0, 1],
        "train_frames": 200,
        "split_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "episodes": [
            {
                "episode_id": i,
                "raw_episode_id": f"fixture-{i}",
                "scene_seed": 0,
                "scene_group_id": "fixture",
                "frames": 100,
                "split": "train",
                "semantic_goal": g,
                "instruction": train[i]["instruction_text"],
            }
            for i, g in enumerate(GOALS)
        ],
    }


def raw_row(index: int) -> dict:
    episode = index % 2
    return {
        "episode_index": torch.tensor(episode),
        "frame_index": torch.tensor(index % 100),
        "task": split()["episodes"][episode]["instruction"],
        "action": torch.tensor([float(index)]),
        "observation.state": torch.zeros(9),
    }


def test_nested_views_preserve_robot_count_and_all_metadata() -> None:
    source = split()
    manifest = make_manifest(source)
    check_manifest(manifest, source)
    assert manifest["validation"]["training_expressions"] == 20
    assert manifest["validation"]["held_out_expressions"] == 12
    for level in (1, 5, 10):
        view = manifest["views"][f"L{level}"]
        assert view["unique_robot_trajectories"] == 2 and view["unique_robot_frames"] == 200
        assert view["label_realizations"] == 2 * level
        assert {r["episode_id"] for r in view["episode_realizations"]} == {0, 1}
        assert all(
            {
                "semantic_goal_id",
                "instruction_text",
                "paraphrase_family",
                "template_id",
                "scene_seed",
                "trajectory_source",
            }
            <= set(r)
            for r in view["episode_realizations"]
        )
    altered = copy.deepcopy(manifest)
    altered["views"]["L5"]["episode_realizations"][0]["scene_seed"] = 999
    with pytest.raises(ValueError, match="differs"):
        check_manifest(altered, source)
    source["train_episode_ids"].append(0)
    with pytest.raises(ValueError, match="identities"):
        make_manifest(source)


@pytest.mark.parametrize("kind", ["text", "template", "side", "m4b"])
def test_leakage_wrong_side_and_original_heldout_rejected(kind: str) -> None:
    train, held = catalogs()
    if kind == "text":
        held[0]["instruction_text"] = "  " + train[0]["instruction_text"].upper().replace(
            ".", "!!!"
        )
    elif kind == "template":
        held[0]["template_id"] = train[1]["template_id"]
    elif kind == "side":
        train[0]["instruction_text"] = train[0]["instruction_text"].replace("left", "right")
    else:
        train[0]["instruction_text"] = "Place the red cube inside the left bin."
    with pytest.raises(ValueError):
        validate_catalogs(train, held)
    assert normalize(" LEFT-hand,  bin! ") == normalize("left hand bin")


def test_sampling_is_repeatable_and_does_not_consume_torch_rng() -> None:
    m = make_manifest(split())
    before = torch.get_rng_state().clone()
    a = [sample_expression(m, "L10", GOALS[i % 2], i, i % 2, i % 100) for i in range(160)]
    b = [sample_expression(m, "L10", GOALS[i % 2], i, i % 2, i % 100) for i in range(160)]
    assert a == b and torch.equal(before, torch.get_rng_state())
    assert len({r["template_id"] for r in a}) > 5


def test_wrapper_changes_only_text_and_excludes_unused_lookahead(tmp_path: Path) -> None:
    source = [raw_row(i) for i in range(24)]
    samples = LanguageSamples(tmp_path / "samples.csv", make_manifest(split()), "L5", split())
    try:
        wrapped = LanguageDataset(source, samples)
        rows = wrapped.__getitems__(list(range(24)))
        for old, new in zip(source, rows, strict=True):
            assert set(old) == set(new)
            assert (
                old["action"] is new["action"]
                and old["observation.state"] is new["observation.state"]
            )
            assert old["task"] == split()["episodes"][int(old["episode_index"])]["instruction"]
        samples.consume(8)
        samples.consume(8)
        receipt = samples.receipt()
        assert receipt["sample_presentations"] == 16 and receipt["unused_prefetched_samples"] == 8
    finally:
        samples.close()
    with (tmp_path / "samples.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 16


def test_resume_consolidation_preserves_failed_tail_and_original_initialization(
    tmp_path: Path,
) -> None:
    m, s = make_manifest(split()), split()
    for attempt, start, count in ((0, 0, 24), (2, 8, 8)):
        path = tmp_path / "language_invocations" / f"attempt-{attempt:02d}"
        samples = LanguageSamples(path / "samples.csv", m, "L10", s, start)
        for i in range(start, start + count):
            samples.relabel(raw_row(i))
        for _ in range(count // 8):
            samples.consume(8)
        samples.close()
        write_json(
            path / "initial_model.json",
            {"start_step": start // 8, "model_state_sha256": str(attempt) * 64},
        )
        write_json(
            path / "runtime.json", {"seconds": 1.5, "completed_optimizer_updates": count // 8}
        )
    empty = tmp_path / "language_invocations/attempt-01"
    write_json(empty / "runtime.json", {"seconds": 0.5, "completed_optimizer_updates": 0})
    original = tmp_path / "language_invocations/attempt-00/samples.csv"
    before = file_digest(original)
    receipt = consolidate_samples(tmp_path, m, "L10", s, 2)
    assert receipt["sample_presentations"] == 16
    assert receipt["initial_model_state_sha256"] == "0" * 64
    assert receipt["total_training_attempt_seconds"] == 3.5
    assert file_digest(original) == before  # abandoned rows are never truncated
    with (tmp_path / "samples-verified.csv").open(newline="") as f:
        assert [int(r["ordinal"]) for r in csv.DictReader(f)] == list(range(16))


def test_model_hash_reads_bfloat16_without_changing_weights_or_rng() -> None:
    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    before = torch.get_rng_state().clone()
    a = model_state_hash(model)
    assert a == model_state_hash(model) and torch.equal(before, torch.get_rng_state())
    with torch.no_grad():
        model.bias.add_(1)
    assert a != model_state_hash(model)


def zero_trace() -> dict[str, np.ndarray]:
    return {
        "tcp": np.array([[-0.3, 0, 0.4]]),
        "cubes": np.array([[[-0.12, 0, 0.025]] * 3]),
        "bins": np.array([[[0.08, 0.18, 0.008], [0.08, -0.18, 0.008]]]),
        "finger_forces": np.zeros((1, 5, 2, 3)),
        "grasped": np.zeros((1, 3), dtype=bool),
    }


def test_new_schedule_scoring_pairs_and_blank_reuse() -> None:
    m = make_manifest(split())
    schedule = make_schedule(m, {"scenes": [{"seed": 123}], "schedule_sha256": "3" * 64})
    assert schedule["physical_rollouts"] == 9 and schedule["scored_rows"] == 12
    rows = []
    for p in schedule["scenes"][0]["prompts"]:
        rows.append(
            {
                **p,
                **diagnostics(zero_trace()),
                "seed": 123,
                "rollout_id": p["prompt_id"],
                "selected_goal": p["prompt_goal"],
                "first_approached_goal": p["prompt_goal"],
                "achieved_goal": p["prompt_goal"],
                "length": 10 if p["prompt_goal"] else 200,
                "video_frames": 11 if p["prompt_goal"] else 201,
                "diagnostic_samples": 11 if p["prompt_goal"] else 201,
                "termination_reason": "goal_reached" if p["prompt_goal"] else "timeout",
                "final_flags": {g: {"success": g == p["prompt_goal"]} for g in GOALS},
                "error": None,
                **dict.fromkeys(
                    (
                        "initial_state_sha256",
                        "initial_rgb_sha256",
                        "initial_qpos_sha256",
                        "noise_schedule_sha256",
                    ),
                    "fixture",
                ),
            }
        )
    result = metrics(rows)
    assert len(scored_rows(rows)) == 12
    assert result["conditions"]["swapped"]["requested_goal"]["full_success"]["count"] == 0
    assert result["conditions"]["swapped"]["supplied_goal"]["full_success"]["count"] == 2
    blank = result["conditions"]["blank"]
    assert blank["physical_rollouts"] == 1 and blank["scored_rows"] == 2
    assert blank["paired_instruction_switch"] is None
    assert blank["supplied_goal"]["full_success"]["rate"] is None
    assert (
        result["conditions"]["seen"]["paired_instruction_switch"][
            "paired_instruction_switch_accuracy"
        ]["count"]
        == 1
    )
    rows[0]["initial_rgb_sha256"] = "different"
    with pytest.raises(ValueError, match="paired"):
        metrics(rows)


def test_zero_action_failure_does_not_fabricate_a_physical_step() -> None:
    value = diagnostics(zero_trace())
    assert value["selected_goal"] is None and value["final_goal"] is None
    assert value["tail_tcp_path_m"] is None and not value["red_contacted"]
    assert all(event["step"] is None for event in value["events"].values())


def test_trajectory_coherence_rejects_early_timeouts_and_false_success() -> None:
    row = {
        "length": 0,
        "termination_reason": "invalid_action",
        "video_frames": 1,
        "diagnostic_samples": 1,
        "achieved_goal": None,
        "final_flags": {g: {"success": False} for g in GOALS},
        "error": None,
    }
    validate_rollout(row)
    for changes in (
        {"termination_reason": "timeout"},
        {"video_frames": 2},
        {"diagnostic_samples": 2},
        {"achieved_goal": GOALS[0]},
        {"error": "native failure"},
    ):
        with pytest.raises(ValueError, match="incoherent"):
            validate_rollout({**row, **changes})


def test_cli_source_config_roundtrips_serialized_paths(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "m4c_cli_fixture", Path(__file__).resolve().parents[2] / "scripts/m4c_baseline.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    # Only the filesystem/config boundary is under test; these mocked gates are not data evidence.
    source = {
        "train_episode_ids": list(range(96)),
        "train_frames": 17149,
        "dataset_total_frames": 64548,
        "all_source_scene_seeds": [],
    }
    schedule = {"excluded_m4a_seeds": [], "schedule_sha256": "fixture"}
    episodes = [{"fixture": True}]
    write_json(tmp_path / "schedule.json", schedule)
    write_json(tmp_path / "gate/schedule.json", schedule)
    write_json(tmp_path / "gate/episodes.json", {"episodes": episodes})
    write_json(
        tmp_path / "gate/gate.json",
        {
            "passed": True,
            "schedule_sha256": "fixture",
            "episodes_sha256": digest(episodes),
        },
    )
    monkeypatch.setattr(cli, "check_split", lambda *args: (source, None))
    monkeypatch.setattr(
        cli,
        "load_manifest",
        lambda *_: SimpleNamespace(config=SimpleNamespace(mode=SimpleNamespace(value="full"))),
    )
    monkeypatch.setattr(cli, "m4b_schedule", lambda *args: schedule)
    monkeypatch.setattr(cli, "verify_assets", lambda path: {"fixture_path": str(path)})
    result = cli.source_data(
        {
            "dataset_root": str(tmp_path),
            "split": str(tmp_path / "split.json"),
            "validation": str(tmp_path / "validation.json"),
            "m4b_schedule": str(tmp_path / "schedule.json"),
            "gate": str(tmp_path / "gate"),
            "assets": str(tmp_path / "assets.json"),
        }
    )
    assert result[0] == source and result[1] == schedule
