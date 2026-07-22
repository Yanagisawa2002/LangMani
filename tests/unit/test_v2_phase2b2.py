from __future__ import annotations

import json
from pathlib import Path

import pytest

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, PANDA_ACTION_COMPONENTS
from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS
from langmani.v2 import phase2b2, push_collection
from langmani.v2.phase2b2 import (
    PHASE2B2_DATASET_ID,
    AcceptedDatasetPackage,
    build_replication_report,
    create_deterministic_tar_gz,
    file_tree_manifest,
    restore_validate_archive,
    write_json_once,
)
from langmani.v2.push_archive import sha256_file
from langmani.v2.push_dataset import (
    PushCollectionConfig,
    PushDatasetContractError,
    build_collection_schedule,
)
from scripts.langmani_v2.evaluate_push_expert import (
    _probe_stop_reason,
)
from scripts.langmani_v2.evaluate_push_expert import (
    _schedule as build_expert_probe_schedule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase_2b2" / "dataset_contract.yaml"
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def _package(**overrides: object) -> AcceptedDatasetPackage:
    values: dict[str, object] = {
        "dataset_id": PHASE2B2_DATASET_ID,
        "dataset_version": "v2",
        "source_commit": "1" * 40,
        "environment_identity": "LangMani-PushToRegion-v0",
        "raw_manifest_sha256": DIGEST_A,
        "lerobot_tree_digest": DIGEST_B,
        "meta_info_sha256": DIGEST_A,
        "split_manifest_sha256": DIGEST_B,
        "raw_archive_sha256": DIGEST_A,
        "lerobot_archive_sha256": DIGEST_B,
        "episode_count": 400,
        "frame_count": 50_000,
        "total_bytes": 123,
        "fps": 20.0,
        "camera_keys": (IMAGE_FEATURE_KEY,),
        "state_dimension": 9,
        "action_dimension": 8,
        "lerobot_version": "0.6.0",
        "acceptance_status": "ACCEPTED",
    }
    values.update(overrides)
    return AcceptedDatasetPackage(**values)  # type: ignore[arg-type]


def test_phase2b2_contract_uses_independent_identity_and_exact_features() -> None:
    config = PushCollectionConfig.load(CONFIG)

    assert config.payload["dataset_version"] == "v2"
    assert config.payload["expert_id"] == "PushToRegionExpert/CandidateP"
    assert config.payload["accepted_expert_commit"] == ("de9309355f96065aa62823522a0d0ebeab7b903c")
    assert config.payload["supersedes_dataset_id"] == "langmani/phase2b-push-v1"
    assert config.payload["state_names"] == list(PANDA_POLICY_STATE_COMPONENTS)
    assert config.payload["action_names"] == list(PANDA_ACTION_COMPONENTS)
    assert config.payload["minimum_accepted_episodes"] == 400
    assert config.payload["minimum_total_frames"] == 50_000
    assert config.payload["atomic_episode_commits"] is True
    assert config.payload["dataset"]["repo_id"] == PHASE2B2_DATASET_ID  # type: ignore[index]


def test_phase2b2_contract_rejects_v1_identity(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["repo_id"] = "langmani/phase2b-push-v1"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PushDatasetContractError, match="independent v2"):
        PushCollectionConfig.load(path)


def test_expert_probe_schedule_is_explicit_and_does_not_consume_formal_seeds() -> None:
    config = PushCollectionConfig.load(CONFIG)
    formal = build_expert_probe_schedule(config)
    probe = build_expert_probe_schedule(
        config,
        seed_start=66_200,
        standard_episodes=8,
        hard_episodes=8,
    )

    assert len(formal) == 100
    assert len(probe) == 16
    assert [seed for seed, _task in probe] == list(range(66_200, 66_216))
    assert {task.difficulty for _seed, task in probe[:8]} == {"standard"}
    assert {task.difficulty for _seed, task in probe[8:]} == {"hard"}
    assert {seed for seed, _task in formal}.isdisjoint(seed for seed, _task in probe)


@pytest.mark.parametrize(
    ("seed_start", "standard_episodes", "hard_episodes", "message"),
    [
        (-1, 1, 0, "non-negative"),
        (1, -1, 1, "non-negative"),
        (1, 0, 0, "cannot be empty"),
    ],
)
def test_expert_probe_schedule_rejects_invalid_bounds(
    seed_start: int,
    standard_episodes: int,
    hard_episodes: int,
    message: str,
) -> None:
    config = PushCollectionConfig.load(CONFIG)

    with pytest.raises(ValueError, match=message):
        build_expert_probe_schedule(
            config,
            seed_start=seed_start,
            standard_episodes=standard_episodes,
            hard_episodes=hard_episodes,
        )


def test_diagnostic_probe_stops_only_when_formal_success_is_impossible() -> None:
    assert (
        _probe_stop_reason(
            completed=20,
            total=64,
            successes=17,
            minimum_success_rate=0.95,
            zero_tolerance_failure=False,
        )
        is None
    )
    assert (
        _probe_stop_reason(
            completed=20,
            total=64,
            successes=16,
            minimum_success_rate=0.95,
            zero_tolerance_failure=False,
        )
        == "success_ceiling_below_gate"
    )
    assert (
        _probe_stop_reason(
            completed=1,
            total=64,
            successes=1,
            minimum_success_rate=0.95,
            zero_tolerance_failure=True,
        )
        == "zero_tolerance_failure"
    )


def test_accepted_package_rejects_v1_and_undersized_data() -> None:
    with pytest.raises(PushDatasetContractError, match="independent v2"):
        _package(dataset_id="langmani/phase2b-push-v1")
    with pytest.raises(PushDatasetContractError, match="undersized"):
        _package(episode_count=399)
    with pytest.raises(PushDatasetContractError, match="undersized"):
        _package(frame_count=49_999)


def test_accepted_package_round_trip_is_complete() -> None:
    package = _package()

    assert AcceptedDatasetPackage.from_dict(package.to_dict()) == package
    assert package.to_dict()["acceptance_status"] == "ACCEPTED"


def test_tree_digest_is_path_independent_and_detects_tampering(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "a.bin").write_bytes(b"same")

    before = file_tree_manifest(left)
    assert before["tree_digest"] == file_tree_manifest(right)["tree_digest"]
    (right / "nested" / "a.bin").write_bytes(b"changed")
    assert before["tree_digest"] != file_tree_manifest(right)["tree_digest"]


def test_tree_manifest_rejects_symlinks_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "target"
    target.write_text("value", encoding="utf-8")
    link = root / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("current Windows identity cannot create symlinks")

    with pytest.raises(PushDatasetContractError, match="symlinks"):
        file_tree_manifest(root)


def test_deterministic_archive_restores_exact_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.txt").write_text("one\n", encoding="utf-8")
    (source / "two.bin").write_bytes(b"\x00\x01")
    tree = file_tree_manifest(source)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_report = create_deterministic_tar_gz(
        source_root=source, archive_path=first, prefix="dataset"
    )
    second_report = create_deterministic_tar_gz(
        source_root=source, archive_path=second, prefix="dataset"
    )
    assert first_report["sha256"] == second_report["sha256"]
    restored = restore_validate_archive(
        archive_path=first,
        destination=tmp_path / "restored",
        prefix="dataset",
        expected_tree_digest=str(tree["tree_digest"]),
    )
    assert restored["passed"] is True
    assert restored["restored_tree_digest"] == tree["tree_digest"]


def test_archive_rejects_unsafe_prefix(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one").write_bytes(b"1")

    with pytest.raises(PushDatasetContractError, match="safe relative"):
        create_deterministic_tar_gz(
            source_root=source,
            archive_path=tmp_path / "bad.tar.gz",
            prefix="../escape",
        )


def test_replication_requires_two_restored_and_three_declared() -> None:
    report = build_replication_report(
        expected_sha256=DIGEST_A,
        replicas=[
            {
                "location": "remote-primary",
                "copy_completed": True,
                "file_size": 1,
                "sha256": DIGEST_A,
                "restore_test_passed": True,
            },
            {
                "location": "local-restored",
                "copy_completed": True,
                "file_size": 1,
                "sha256": DIGEST_A,
                "restore_test_passed": True,
            },
            {
                "location": "downloadable-pending",
                "copy_completed": False,
                "file_size": 0,
                "sha256": None,
                "restore_test_passed": False,
            },
        ],
    )
    assert report["minimum_temporary_persistence_gate"] is True
    assert report["three_independent_copies"] is False
    assert report["passed"] is True

    incomplete = build_replication_report(
        expected_sha256=DIGEST_A,
        replicas=[
            {
                "location": "only-copy",
                "copy_completed": True,
                "file_size": 1,
                "sha256": DIGEST_A,
                "restore_test_passed": True,
            }
        ],
    )
    assert incomplete["passed"] is False


def test_replication_rejects_hash_drift() -> None:
    with pytest.raises(PushDatasetContractError, match="hash changed"):
        build_replication_report(
            expected_sha256=DIGEST_A,
            replicas=[
                {
                    "location": "bad-copy",
                    "copy_completed": True,
                    "file_size": 1,
                    "sha256": DIGEST_B,
                    "restore_test_passed": False,
                }
            ],
        )


def test_owned_partial_cleanup_is_narrow(tmp_path: Path) -> None:
    parent = tmp_path / "shard"
    partial = parent / ".attempt-0001.partial"
    partial.mkdir(parents=True)
    (partial / "partial.bin").write_bytes(b"partial")
    push_collection._remove_owned_partial(partial, parent)
    assert not partial.exists()

    unowned = parent / "someone-else"
    unowned.mkdir()
    with pytest.raises(PushDatasetContractError, match="unowned"):
        push_collection._remove_owned_partial(unowned, parent)
    assert unowned.is_dir()


def test_atomic_shard_resume_reuses_verified_episode_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    config = PushCollectionConfig.load(CONFIG)
    scheduled = build_collection_schedule(config, "full")[0]
    attempts = tmp_path / "raw" / "attempts" / "full"
    episode = attempts / "shard-0000" / "attempt-0000"
    episode.mkdir(parents=True)
    h5_path = episode / "trajectory.h5"
    json_path = episode / "trajectory.json"
    h5_path.write_bytes(b"h5")
    json_path.write_bytes(b"json")
    relative_h5 = h5_path.relative_to(tmp_path / "raw").as_posix()
    relative_json = json_path.relative_to(tmp_path / "raw").as_posix()
    record = {"episode_id": scheduled.episode_id}
    archive = {
        "episode_id": scheduled.episode_id,
        "h5_path": relative_h5,
        "json_path": relative_json,
        "h5_sha256": sha256_file(h5_path),
        "json_sha256": sha256_file(json_path),
    }
    (episode / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "langmani-v2-phase2b2-atomic-episode-v0",
                "episode_id": scheduled.episode_id,
                "schedule": scheduled.to_dict(),
                "record": record,
                "archive": archive,
                "completed": True,
            }
        ),
        encoding="utf-8",
    )
    (attempts / "shard-0000" / "manifest.json").write_text(
        json.dumps({"completed": True, "attempt_records": [record]}), encoding="utf-8"
    )

    resumed = push_collection._collect_shard_atomic(
        [scheduled],
        shard_index=0,
        attempts_root=attempts,
        run_id="run",
        sim_backend="physx_cuda",
        resume=True,
        expert_identity="PushToRegionExpert/CandidateN@f04cd60",
    )
    assert resumed == [record]

    h5_path.write_bytes(b"tampered")
    with pytest.raises(PushDatasetContractError, match="byte identity changed"):
        push_collection._collect_shard_atomic(
            [scheduled],
            shard_index=0,
            attempts_root=attempts,
            run_id="run",
            sim_backend="physx_cuda",
            resume=True,
            expert_identity="PushToRegionExpert/CandidateN@f04cd60",
        )


def test_source_paths_cannot_escape_raw_root(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()

    assert phase2b2._source_path(root, "attempts/a.h5") == root / "attempts" / "a.h5"
    with pytest.raises(PushDatasetContractError, match="escapes"):
        phase2b2._source_path(root, "../outside.h5")


def test_write_json_once_is_idempotent_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_json_once(path, {"optimizer_steps": 0, "training_started": False})
    write_json_once(path, {"optimizer_steps": 0, "training_started": False})

    with pytest.raises(PushDatasetContractError, match="overwrite"):
        write_json_once(path, {"optimizer_steps": 1, "training_started": True})
