"""CPU-safe tests for the non-bypassable M3B source boundary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

import langmani.datasets.lerobot_source as source_module
from langmani.datasets.archive import sha256_file
from langmani.datasets.identity import COLLECTION_SCHEMA_VERSION
from langmani.datasets.lerobot_source import (
    M3ASourceValidationError,
    RawEpisodeReadError,
    ValidatedM3ASource,
    ValidatedRawEpisodeReader,
    validate_m3a_source_for_export,
)
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.datasets.types import CollectionStatus, ReplayValidationMode
from langmani.environments.specs import canonical_instruction, stable_scene_id, stable_task_id

_CONFIG_FINGERPRINT = "sha256:" + "1" * 64
_SOURCE_DIGEST = "sha256:" + "2" * 64
_COLLECTION_RUN_ID = "collection-run-test"
_SUCCESS_EVALUATION = {
    "success": True,
    "fail": False,
    "target_in_wrong_bin": False,
    "wrong_object_in_target_bin": False,
}


def _write_raw_pair(root: Path, *, transition_count: int = 2) -> tuple[Path, Path, np.ndarray]:
    h5_path = root / "accepted" / "shards" / "source.h5"
    json_path = h5_path.with_suffix(".json")
    h5_path.parent.mkdir(parents=True)
    actions = np.arange(transition_count * 8, dtype=np.float32).reshape(transition_count, 8)
    with h5py.File(h5_path, "w") as archive:
        trajectory = archive.create_group("traj_0")
        trajectory.create_dataset("actions", data=actions)
        states = trajectory.create_group("env_states")
        actors = states.create_group("actors")
        actors.create_dataset(
            "red_cube",
            data=np.arange((transition_count + 1) * 13, dtype=np.float32).reshape(
                transition_count + 1, 13
            ),
        )
        articulations = states.create_group("articulations")
        articulations.create_dataset(
            "panda",
            data=np.arange((transition_count + 1) * 40, dtype=np.float32).reshape(
                transition_count + 1, 40
            ),
        )
    json_path.write_text('{"fixture":true}\n', encoding="utf-8")
    return h5_path, json_path, actions


def _fake_source_contract(
    root: Path,
    *,
    group_count: int,
    h5_path: Path | None = None,
    json_path: Path | None = None,
    h5_sha256: str | None = None,
    json_sha256: str | None = None,
    elapsed_steps: int = 2,
) -> tuple[Any, Any]:
    relative_h5 = (
        h5_path.relative_to(root).as_posix() if h5_path is not None else "accepted/shards/source.h5"
    )
    relative_json = (
        json_path.relative_to(root).as_posix()
        if json_path is not None
        else "accepted/shards/source.json"
    )
    h5_digest = h5_sha256 or "a" * 64
    json_digest = json_sha256 or "b" * 64
    episodes: list[Any] = []
    groups: list[Any] = []
    for group_index in range(group_count):
        group_episodes: list[Any] = []
        for task_index, task_spec in enumerate(CANONICAL_TASK_SPECS):
            flat_index = group_index * 6 + task_index
            episode = SimpleNamespace(
                scheduled_episode_id=f"scheduled-{flat_index}",
                attempt_id=f"attempt-{flat_index}",
                raw_trajectory_id=f"raw-{flat_index}",
                source_shard_id="source-shard-0",
                native_episode_id=flat_index,
                h5_group=f"traj_{flat_index}",
                h5_path=relative_h5,
                json_path=relative_json,
                h5_sha256=h5_digest,
                json_sha256=json_digest,
                elapsed_steps=elapsed_steps,
                scene_seed=group_index,
                scene_id=stable_scene_id(group_index),
                task_spec=task_spec,
                task_id=stable_task_id(task_spec),
                canonical_instruction=canonical_instruction(task_spec),
                expert_result=SimpleNamespace(
                    success=True,
                    total_environment_steps=elapsed_steps,
                ),
                replay_validation=SimpleNamespace(
                    passed=True,
                    mode=ReplayValidationMode.ACTION_AND_STATE_AUDIT,
                    recorded_success=True,
                    replay_success=True,
                    task_spec_matches=True,
                    wrong_object_in_target_bin=False,
                    target_in_wrong_bin=False,
                    target_off_table=False,
                    recorded_action_steps=elapsed_steps,
                    replayed_action_steps=elapsed_steps,
                    state_audit_performed=True,
                    state_audit_passed=True,
                    failure_codes=(),
                    failure_reasons=(),
                ),
                final_environment_evaluation=dict(_SUCCESS_EVALUATION),
                structural_valid=True,
                accepted=True,
            )
            group_episodes.append(episode)
            episodes.append(episode)
        groups.append(
            SimpleNamespace(
                candidate_scene_index=group_index,
                candidate_scene_id=f"candidate-{group_index}",
                accepted_scene_group_id=f"accepted-group-{group_index}",
                scene_seed=group_index,
                scene_id=stable_scene_id(group_index),
                scheduled_episode_ids=tuple(
                    episode.scheduled_episode_id for episode in group_episodes
                ),
                episodes=tuple(group_episodes),
                complete=True,
                accepted=True,
            )
        )
    shard = SimpleNamespace(
        source_shard_id="source-shard-0",
        shard_index=0,
        h5_path=relative_h5,
        json_path=relative_json,
        h5_sha256=h5_digest,
        json_sha256=json_digest,
        h5_size_bytes=h5_path.stat().st_size if h5_path is not None else 1,
        json_size_bytes=json_path.stat().st_size if json_path is not None else 1,
        raw_trajectory_ids=tuple(episode.raw_trajectory_id for episode in episodes),
    )
    task_counts = Counter(episode.task_id for episode in episodes)
    manifest = SimpleNamespace(
        collection_schema_version=COLLECTION_SCHEMA_VERSION,
        collection_run_id=_COLLECTION_RUN_ID,
        status=CollectionStatus.COMPLETE,
        schedule=SimpleNamespace(config_fingerprint=_CONFIG_FINGERPRINT),
        config=SimpleNamespace(
            replay_validation_mode=ReplayValidationMode.ACTION_AND_STATE_AUDIT,
            target_complete_scene_count=group_count,
        ),
        scene_groups=tuple(groups),
        raw_episodes=tuple(episodes),
        source_shards=(shard,),
    )
    summary = SimpleNamespace(
        collection_schema_version=COLLECTION_SCHEMA_VERSION,
        status=CollectionStatus.COMPLETE,
        accepted_scene_group_count=group_count,
        accepted_episode_count=group_count * 6,
        task_episode_counts=task_counts,
    )
    return manifest, summary


def _verification_payload(root: Path, *, mode: str, digest: str = _SOURCE_DIGEST) -> dict[str, Any]:
    target_mode = "target_smoke" if mode == "smoke" else "target_full"
    return {
        "schema_version": "langmani-m3a-verification-v3",
        "verification_mode": target_mode,
        "validation_scope": target_mode,
        "target_mode": True,
        "dataset_root": str(root.resolve()),
        "implementation_validated": True,
        "prior_target_gates_validated": True,
        "expert_collection_smoke_validated": mode == "smoke",
        "action_replay_validated": True,
        "full_dataset_validated": mode == "full",
        "physical_target_validated": True,
        "physical_acceptance": True,
        "passed": True,
        "source_archive_identity_status": "validated",
        "source_archive_identity_error": None,
        "collection_run_id": _COLLECTION_RUN_ID,
        "config_fingerprint": _CONFIG_FINGERPRINT,
        "source_archive_digest": digest,
    }


def _install_gate_fakes(
    monkeypatch: pytest.MonkeyPatch,
    manifest: Any,
    summary: Any,
    *,
    digest: str = _SOURCE_DIGEST,
) -> None:
    monkeypatch.setattr(source_module, "inspect_raw_dataset", lambda root: summary)
    monkeypatch.setattr(source_module, "load_manifest", lambda root: manifest)
    monkeypatch.setattr(source_module, "source_archive_digest", lambda value: digest)


def _gate(
    root: Path,
    report_path: Path,
    *,
    mode: str,
) -> ValidatedM3ASource:
    return validate_m3a_source_for_export(
        root,
        verification_report_path=report_path,
        mode=mode,  # type: ignore[arg-type]
        expected_run_fingerprint=_CONFIG_FINGERPRINT,
        expected_schema_version=COLLECTION_SCHEMA_VERSION,
    )


def test_inspector_failure_prevents_manifest_and_hdf5_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fail_inspection(root: Path) -> None:
        events.append("inspect")
        raise RuntimeError("corrupt source")

    monkeypatch.setattr(source_module, "inspect_raw_dataset", fail_inspection)
    monkeypatch.setattr(
        source_module,
        "load_manifest",
        lambda root: events.append("manifest") or object(),
    )
    monkeypatch.setattr(
        source_module.h5py,
        "File",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("HDF5 must stay closed")),
    )

    with pytest.raises(M3ASourceValidationError, match="source inspection failed"):
        _gate(tmp_path / "raw", tmp_path / "verification.json", mode="smoke")

    assert events == ["inspect"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"schema_version": "langmani-m3a-verification-v2"}, "schema"),
        ({"action_replay_validated": False}, "target evidence"),
        ({"source_archive_digest": "sha256:" + "9" * 64}, "digest is stale"),
        ({"verification_mode": "target_full"}, "mode differs"),
        ({"dataset_root": "__OTHER_ROOT__"}, "different source root"),
    ],
)
def test_gate_rejects_stale_or_mismatched_v3_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any],
    match: str,
) -> None:
    root = (tmp_path / "raw").resolve()
    root.mkdir()
    manifest, summary = _fake_source_contract(root, group_count=1)
    _install_gate_fakes(monkeypatch, manifest, summary)
    payload = _verification_payload(root, mode="smoke")
    payload.update(mutation)
    if payload["dataset_root"] == "__OTHER_ROOT__":
        payload["dataset_root"] = str((tmp_path / "other").resolve())
    report_path = tmp_path / "verification.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(M3ASourceValidationError, match=match):
        _gate(root, report_path, mode="smoke")


def test_smoke_and_full_gates_use_canonical_balanced_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mode, group_count in (("smoke", 1), ("full", 60)):
        root = (tmp_path / mode).resolve()
        root.mkdir()
        manifest, summary = _fake_source_contract(root, group_count=group_count)
        _install_gate_fakes(monkeypatch, manifest, summary)
        report_path = tmp_path / f"{mode}.json"
        report_path.write_text(
            json.dumps(_verification_payload(root, mode=mode)),
            encoding="utf-8",
        )

        source = _gate(root, report_path, mode=mode)

        assert len(source.ordered_groups) == group_count
        assert len(source.ordered_episodes) == group_count * 6
        assert (
            tuple(episode.task_spec for episode in source.ordered_episodes[:6])
            == CANONICAL_TASK_SPECS
        )
        assert set(Counter(episode.task_id for episode in source.ordered_episodes).values()) == {
            group_count
        }


def test_gate_rejects_noncanonical_or_non_bijective_accepted_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "raw").resolve()
    root.mkdir()
    manifest, summary = _fake_source_contract(root, group_count=1)
    report_path = tmp_path / "verification.json"
    report_path.write_text(
        json.dumps(_verification_payload(root, mode="smoke")),
        encoding="utf-8",
    )

    first_group = manifest.scene_groups[0]
    manifest.scene_groups = (
        SimpleNamespace(
            **{
                **vars(first_group),
                "episodes": (
                    first_group.episodes[1],
                    first_group.episodes[0],
                    *first_group.episodes[2:],
                ),
                "scheduled_episode_ids": (
                    first_group.scheduled_episode_ids[1],
                    first_group.scheduled_episode_ids[0],
                    *first_group.scheduled_episode_ids[2:],
                ),
            }
        ),
    )
    _install_gate_fakes(monkeypatch, manifest, summary)
    with pytest.raises(M3ASourceValidationError, match="canonical order"):
        _gate(root, report_path, mode="smoke")

    manifest, summary = _fake_source_contract(root, group_count=1)
    manifest.raw_episodes = (*manifest.raw_episodes, manifest.raw_episodes[0])
    _install_gate_fakes(monkeypatch, manifest, summary)
    with pytest.raises(M3ASourceValidationError, match="one-to-one"):
        _gate(root, report_path, mode="smoke")


@pytest.mark.parametrize("failure_kind", ["expert", "replay"])
def test_gate_rejects_failure_evidence_inside_an_accepted_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    root = (tmp_path / "raw").resolve()
    root.mkdir()
    manifest, summary = _fake_source_contract(root, group_count=1)
    episode = manifest.raw_episodes[0]
    if failure_kind == "expert":
        episode.expert_result.success = False
    else:
        episode.replay_validation.passed = False
    _install_gate_fakes(monkeypatch, manifest, summary)
    report_path = tmp_path / "verification.json"
    report_path.write_text(
        json.dumps(_verification_payload(root, mode="smoke")),
        encoding="utf-8",
    )

    with pytest.raises(M3ASourceValidationError, match="failure evidence"):
        _gate(root, report_path, mode="smoke")


def test_reader_preserves_actions_and_t_plus_one_states_without_terminal_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "raw").resolve()
    h5_path, json_path, expected_actions = _write_raw_pair(root)
    manifest, summary = _fake_source_contract(
        root,
        group_count=1,
        h5_path=h5_path,
        json_path=json_path,
        h5_sha256=sha256_file(h5_path),
        json_sha256=sha256_file(json_path),
    )
    _install_gate_fakes(monkeypatch, manifest, summary)
    report_path = tmp_path / "verification.json"
    report_path.write_text(
        json.dumps(_verification_payload(root, mode="smoke")),
        encoding="utf-8",
    )
    source = _gate(root, report_path, mode="smoke")
    real_sha256_file = sha256_file
    checksum_calls: list[Path] = []

    def counted_sha256(path: str | Path) -> str:
        checksum_calls.append(Path(path))
        return real_sha256_file(path)

    monkeypatch.setattr(source_module, "sha256_file", counted_sha256)
    reader = ValidatedRawEpisodeReader(source)
    episode = reader.read_episode("raw-0")
    reader.read_episode("raw-0")

    assert np.array_equal(episode.actions, expected_actions)
    assert episode.actions.dtype == np.float32
    assert not episode.actions.flags.writeable
    assert episode.transition_count == 2
    assert episode.state_count == 3
    assert episode.terminal_state_index == 2
    assert float(episode.state_at(2)["actors"]["red_cube"][0]) == 26.0
    pairs = tuple(episode.iter_training_pairs())
    assert len(pairs) == 2
    assert np.array_equal(pairs[0][1], expected_actions[0])
    assert np.array_equal(pairs[1][1], expected_actions[1])
    with pytest.raises(IndexError, match="T-1"):
        episode.training_state_at(2)
    assert checksum_calls == [h5_path, json_path]


def test_reader_rejects_path_escape_and_changed_shard_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "raw").resolve()
    h5_path, json_path, _ = _write_raw_pair(root)
    manifest, summary = _fake_source_contract(
        root,
        group_count=1,
        h5_path=h5_path,
        json_path=json_path,
        h5_sha256=sha256_file(h5_path),
        json_sha256=sha256_file(json_path),
    )
    _install_gate_fakes(monkeypatch, manifest, summary)
    report_path = tmp_path / "verification.json"
    report_path.write_text(
        json.dumps(_verification_payload(root, mode="smoke")),
        encoding="utf-8",
    )
    source = _gate(root, report_path, mode="smoke")

    manifest.source_shards[0].h5_path = "../escaped.h5"
    manifest.raw_episodes[0].h5_path = "../escaped.h5"
    with pytest.raises(RawEpisodeReadError, match="escapes"):
        ValidatedRawEpisodeReader(source).read_episode("raw-0")

    manifest.source_shards[0].h5_path = h5_path.relative_to(root).as_posix()
    manifest.raw_episodes[0].h5_path = manifest.source_shards[0].h5_path
    with h5_path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RawEpisodeReadError, match="checksum changed"):
        ValidatedRawEpisodeReader(source).read_episode("raw-0")


def test_reader_revalidates_source_digest_after_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "raw").resolve()
    root.mkdir()
    manifest, summary = _fake_source_contract(root, group_count=1)
    _install_gate_fakes(monkeypatch, manifest, summary)
    report_path = tmp_path / "verification.json"
    report_path.write_text(
        json.dumps(_verification_payload(root, mode="smoke")),
        encoding="utf-8",
    )
    source = _gate(root, report_path, mode="smoke")
    reader = ValidatedRawEpisodeReader(source)

    reader.revalidate_source()
    monkeypatch.setattr(source_module, "source_archive_digest", lambda value: "sha256:" + "9" * 64)
    with pytest.raises(RawEpisodeReadError, match="digest changed"):
        reader.revalidate_source()


def test_validated_source_cannot_be_constructed_without_gate(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="only be created"):
        ValidatedM3ASource(
            root=tmp_path,
            verification_report_path=tmp_path / "report.json",
            mode="smoke",
            manifest=object(),  # type: ignore[arg-type]
            summary=object(),  # type: ignore[arg-type]
            config_fingerprint=_CONFIG_FINGERPRINT,
            archive_digest=_SOURCE_DIGEST,
            ordered_groups=(),
            ordered_episodes=(),
            _gate_token=object(),
        )
