"""Synthetic contract tests only; no generated fixture is benchmark evidence."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from langmani.policies.m4a_data import IMAGE, STATE, digest, file_digest, write_json
from langmani.policies.m4b_data import check_split, make_split
from langmani.policies.m4b_evaluation import noise_schedule, run_policy, write_video
from langmani.policies.m4b_protocol import GOALS, make_schedule, metrics, prompts, score_rollouts
from langmani.policies.m4b_training import policy_observation, verify_assets


def test_two_goal_split_is_balanced_and_preserves_scene_partition() -> None:
    from test_m4a_baseline import full_manifest

    manifest = full_manifest()
    report = {
        "passed": True,
        "source_id": manifest.export_fingerprint,
        "content_sha256": "fixture",
        "num_frames": 1080,
    }
    split = make_split(manifest, report)
    assert len(split["episodes"]) == 120
    assert [len(split[f"{key}_episode_ids"]) for key in ("train", "held_out", "excluded_test")] == [
        96,
        12,
        12,
    ]
    for key in ("train", "held_out", "excluded_test"):
        records = [r for r in split["episodes"] if r["episode_id"] in split[f"{key}_episode_ids"]]
        assert sum(r["semantic_goal"] == GOALS[0] for r in records) == len(records) // 2
    assert split["split_sha256"] == digest({k: v for k, v in split.items() if k != "split_sha256"})
    with pytest.raises(ValueError, match="incomplete"):
        make_split(
            SimpleNamespace(
                export_fingerprint=manifest.export_fingerprint, episodes=manifest.episodes[1:]
            ),
            report,
        )


def test_byte_bound_validation_refuses_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_m4a_baseline import full_manifest

    import langmani.policies.m4b_data as module

    root = tmp_path / "data"
    root.mkdir()
    (root / "frame").write_bytes(b"fixture")
    inventory = {"frame": file_digest(root / "frame")}
    manifest = full_manifest()
    report = {
        "passed": True,
        "source_id": manifest.export_fingerprint,
        "num_frames": 1080,
        "content_sha256": digest(inventory),
        "files": inventory,
        "m4b_pair_audit": {},
    }
    write_json(tmp_path / "validation.json", report)
    split = make_split(manifest, report)
    write_json(tmp_path / "split.json", split)
    monkeypatch.setattr(module, "load_manifest", lambda p: manifest)
    assert check_split(root, tmp_path / "split.json", tmp_path / "validation.json")[0] == split
    (root / "frame").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        check_split(root, tmp_path / "split.json", tmp_path / "validation.json")


def test_schedule_reproducible_excluded_and_unseen_paraphrases() -> None:
    schedule = make_schedule(list(range(60)), list(range(100, 120)))
    assert schedule == make_schedule(list(range(60)), list(range(100, 120)))
    assert len({s["seed"] for s in schedule["scenes"]}) == 20
    assert not {s["seed"] for s in schedule["scenes"]} & set(range(120))
    for scene in schedule["scenes"]:
        assert len(scene["prompts"]) == 5
        assert scene["prompts"][2]["instruction"] == ""
        assert {p["instruction"] for p in scene["prompts"][:2]}.isdisjoint(
            {p["instruction"] for p in scene["prompts"][3:]}
        )
    with pytest.raises(ValueError):
        make_schedule([], [])


def fixture_rollouts(behavior: str) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts(0):
        achieved = GOALS[0] if behavior == "constant" else prompt["prompt_goal"]
        if behavior == "timeout":
            achieved = None
        rows.append(
            {
                **prompt,
                "seed": 42,
                "rollout_id": prompt["prompt_id"],
                "achieved_goal": achieved,
                "length": 20 if achieved else 200,
                "termination_reason": "goal_reached" if achieved else "timeout",
                "video": "fixture.mp4",
                "video_sha256": "v",
                "initial_state_sha256": "s",
                "initial_rgb_sha256": "i",
                "initial_qpos_sha256": "q",
                "noise_schedule_sha256": "n",
                "actions_sha256": achieved or "a",
                "error": None,
            }
        )
    return rows


def test_metrics_distinguish_constant_destination_from_language_following() -> None:
    constant = metrics(fixture_rollouts("constant"))
    assert constant["conditions"]["correct"]["success_rate"] == 0.5
    assert constant["canonical_pair_goal_switch_success_rate"] == 0
    following = metrics(fixture_rollouts("following"))
    assert following["physical_policy_rollouts"] == 5 and following["scored_rows"] == 8
    assert following["canonical_pair_goal_switch_success_rate"] == 1
    assert following["conditions"]["swapped"]["success_rate"] == 0
    assert following["conditions"]["swapped"]["prompt_goal_success_rate"] == 1
    assert following["conditions"]["correct"]["mean_successful_episode_length"] == 20
    assert following["physical_timeout_count"] == 1
    scored = score_rollouts(fixture_rollouts("following"))
    assert len({r["rollout_id"] for r in scored}) == 5
    assert len([r for r in scored if r["rollout_id"] == "blank"]) == 2
    timeout = metrics(fixture_rollouts("timeout"))
    assert timeout["conditions"]["correct"]["mean_successful_episode_length"] is None
    assert timeout["canonical_paraphrase_nonnull_goal_agreement_rate"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_rgb_sha256", "changed"),
        ("initial_state_sha256", None),
        ("noise_schedule_sha256", "different"),
        ("length", 0),
        ("video_sha256", None),
        ("error", "broken"),
    ],
)
def test_invalid_pairing_cannot_establish_language_use(field: str, value: Any) -> None:
    rows = fixture_rollouts("following")
    rows[0][field] = value
    result = metrics(rows)
    assert not result["passed"] and result["canonical_pair_goal_switch_success_rate"] is None


def test_duplicate_or_missing_rollout_rejected() -> None:
    rows = fixture_rollouts("following")
    for invalid in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
        with pytest.raises(ValueError):
            score_rollouts(invalid)


def test_blank_text_never_replaced_and_policy_allowlist() -> None:
    batch = policy_observation(np.zeros((3, 256, 256), np.uint8), np.zeros(9, np.float32), "")
    assert set(batch) == {IMAGE, STATE, "task"} and batch["task"] == ""
    with pytest.raises(ValueError, match="explicit"):
        policy_observation(np.zeros((3, 256, 256), np.uint8), np.zeros(9, np.float32), None)


def test_common_flow_noise_has_fixed_length_and_seed() -> None:
    first, fingerprint = noise_schedule(7)
    second, same = noise_schedule(7)
    assert fingerprint == same and fingerprint != noise_schedule(8)[1]
    assert len(first) == 20 and first[0].shape == (1, 50, 32)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_model_receipt_rejects_changed_bytes(tmp_path: Path) -> None:
    from langmani.policies.m4b_protocol import (
        BACKBONE_MODEL,
        BACKBONE_REVISION,
        BASE_MODEL,
        BASE_REVISION,
    )

    receipt = {}
    for name, repo, rev in (
        ("base", BASE_MODEL, BASE_REVISION),
        ("backbone", BACKBONE_MODEL, BACKBONE_REVISION),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        receipt[name] = {
            "repo_id": repo,
            "revision": rev,
            "path": str(directory),
            "files": {"config.json": file_digest(directory / "config.json")},
        }
    path = tmp_path / "assets.json"
    write_json(path, receipt)
    assert verify_assets(path) == receipt
    (tmp_path / "base/config.json").write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_assets(path)


@pytest.mark.parametrize("gripper", [1.05, float("nan")])
def test_policy_rollout_scores_other_goal_without_privileged_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gripper: float
) -> None:
    import langmani.datasets.observation_reconstruction as reconstruction
    import langmani.datasets.policy_state as state
    import langmani.policies.m4b_evaluation as evaluation

    monkeypatch.setattr(
        reconstruction, "extract_base_camera_rgb", lambda obs: np.zeros((256, 256, 3), np.uint8)
    )
    monkeypatch.setattr(
        state, "extract_panda_policy_state_v0", lambda robot: np.zeros(9, np.float32)
    )
    steps = []
    monkeypatch.setattr(
        evaluation,
        "score_state",
        lambda env: {
            g: {"success": bool(steps) and g == GOALS[1], "target_off_table": False} for g in GOALS
        },
    )

    class Policy:
        config = SimpleNamespace(device="cpu")
        resets = 0

        def reset(self) -> None:
            self.resets += 1

        def select_action(self, batch: dict[str, Any], noise: torch.Tensor) -> torch.Tensor:
            assert set(batch) == {IMAGE, STATE, "task"} and batch["task"] == ""
            assert noise.shape == (1, 50, 32)
            return torch.tensor([[0.0] * 7 + [gripper]], dtype=torch.float32)

    class Processor:
        def reset(self) -> None:
            pass

        def __call__(self, value: Any) -> Any:
            return value

    def step(action: np.ndarray) -> tuple[Any, ...]:
        steps.append(action)
        return {}, 0, np.array(False), np.array(False), {}

    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            single_action_space=SimpleNamespace(shape=(8,), low=np.full(8, -1), high=np.ones(8)),
            agent=SimpleNamespace(robot=None),
        ),
        step=step,
    )
    policy = Policy()
    result = run_policy(
        env,
        {"privileged": "forbidden"},
        policy,
        Processor(),
        Processor(),
        "",
        42,
        tmp_path / "rollout",
    )
    assert policy.resets == 1
    if np.isfinite(gripper):
        assert result["achieved_goal"] == GOALS[1] and result["length"] == 1
        assert result["gripper_saturation_count"] == 1 and steps[0][-1] == 1
    else:
        assert result["termination_reason"] == "invalid_action" and result["length"] == 0
    assert result["video_frames"] == result["length"] + 1


def test_evidence_video_decodes(tmp_path: Path) -> None:
    write_video(tmp_path / "video.mp4", [np.full((256, 256, 3), i, np.uint8) for i in range(3)])
    assert (tmp_path / "video.mp4").stat().st_size > 0


def test_cli_blocks_writing_into_frozen_m4a(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "smolvla_cli", Path(__file__).parents[2] / "scripts/smolvla_baseline.py"
    )
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(ValueError, match="this checkout"):
        cli.safe_output(tmp_path / "m4a/results/new", tmp_path / "data")
    args = cli.parse_args(
        [
            "train",
            "--dataset-root",
            "data",
            "--output",
            "results/smoke",
            "--validation",
            "v.json",
            "--split",
            "s.json",
            "--schedule",
            "seeds.json",
            "--assets",
            "a.json",
            "--gate",
            "gate",
            "--mode",
            "smoke",
        ]
    )
    assert args.batch_size == 8 and args.learning_rate == 1e-4 and args.seed == 0
