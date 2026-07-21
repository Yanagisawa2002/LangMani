"""Outcome-free validation and sealed-test schedules for Phase 2C."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c import Phase2CContractError
from langmani.v2.push_dataset import assign_split

SCHEDULE_SCHEMA = "langmani-v2-phase2c-evaluation-schedule-v0"


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase2CContractError(f"{label} must be a non-negative integer")
    return value


def _record_key(record: Mapping[str, object]) -> tuple[str, str, str]:
    task = record.get("task_spec")
    if not isinstance(task, Mapping):
        raise Phase2CContractError("episode task_spec is malformed")
    return (
        str(task.get("target_object_id")),
        str(task.get("target_region_id")),
        str(task.get("difficulty")),
    )


def _balanced(records: Sequence[Mapping[str, object]], count: int) -> list[Mapping[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        groups[_record_key(record)].append(record)
    for values in groups.values():
        values.sort(key=lambda item: str(item.get("episode_id")))
    selected: list[Mapping[str, object]] = []
    keys = sorted(groups)
    while len(selected) < min(count, len(records)):
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def _supplemental_scene_group(
    *, split: str, seed: int, task_spec: Mapping[str, object], template_group: str
) -> str:
    for nonce in range(100_000):
        value = f"phase2c-{split}-seed{seed}-scene{nonce}"
        candidate = {
            "template_group": template_group,
            "task_spec": dict(task_spec),
            "scene_group_id": value,
        }
        if assign_split(candidate) == split:
            return value
    raise Phase2CContractError(f"cannot derive a supplemental {split} scene group")


def _entry(
    record: Mapping[str, object],
    *,
    split: str,
    seed: int | None = None,
    scene_group_id: str | None = None,
    source_kind: str = "phase2b_held_out_episode",
) -> dict[str, object]:
    task = record.get("task_spec")
    if not isinstance(task, Mapping):
        raise Phase2CContractError("schedule source lacks task_spec")
    instruction = record.get("instruction")
    if not isinstance(instruction, str) or not instruction:
        raise Phase2CContractError("schedule source lacks instruction")
    resolved_seed = _integer(record["seed"], "seed") if seed is None else seed
    geometry = "cube" if task.get("target_object_id") == "blue_cube" else "horizontal_cylinder"
    semantic = {
        "split": split,
        "seed": resolved_seed,
        "task_spec": dict(task),
        "language_instruction": instruction,
        "template_group": str(record.get("template_group")),
        "template_id": str(record.get("template_id")),
        "scene_group_id": scene_group_id or str(record.get("scene_group_id")),
        "geometry": geometry,
        "direction": str(task.get("target_region_id")),
        "difficulty": str(task.get("difficulty")),
        "visual_domain": "photometric_shift_v0" if split == "test_visual_shift" else "base",
        "source_kind": source_kind,
        "source_episode_id": record.get("episode_id"),
    }
    return {
        **semantic,
        "evaluation_id": f"phase2c:{sha256_hex(semantic)[:24]}",
    }


def build_phase2c_schedules(
    records: Sequence[Mapping[str, object]],
    *,
    development_count: int = 40,
    train_sanity_count: int = 30,
    final_per_split: int = 50,
    supplemental_seed_start: int = 2_300_000,
) -> dict[str, object]:
    """Freeze balanced schedules from split metadata only, before outcomes exist."""

    by_split: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    seen_episode: set[str] = set()
    seen_seed: set[int] = set()
    for record in records:
        episode_id = str(record.get("episode_id"))
        if not episode_id or episode_id in seen_episode:
            raise Phase2CContractError("schedule input has missing or duplicate episode identity")
        split = str(record.get("split"))
        if not split:
            raise Phase2CContractError("schedule input lacks split")
        seen_episode.add(episode_id)
        seen_seed.add(_integer(record["seed"], "seed"))
        by_split[split].append(record)

    validation = _balanced(by_split["validation"], development_count)
    if len(validation) != development_count:
        raise Phase2CContractError("not enough validation identities for development schedule")
    development = [_entry(item, split="validation") for item in validation]
    train = _balanced(by_split["train"], train_sanity_count)
    if len(train) != train_sanity_count:
        raise Phase2CContractError("not enough train identities for sanity schedule")
    sealed: list[dict[str, object]] = [
        _entry(item, split="train_distribution_sanity") for item in train
    ]
    next_seed = supplemental_seed_start
    test_splits = (
        "test_unseen_scene",
        "test_unseen_language",
        "test_hard",
        "test_visual_shift",
    )
    supplemental_counts: Counter[str] = Counter()
    for split in test_splits:
        source = _balanced(by_split[split], len(by_split[split]))
        if not source:
            raise Phase2CContractError(f"held-out split {split} is empty")
        entries = [_entry(item, split=split) for item in source]
        cursor = 0
        while len(entries) < final_per_split:
            template = source[cursor % len(source)]
            task_spec = template["task_spec"]
            assert isinstance(task_spec, Mapping)
            template_group = str(template["template_group"])
            while next_seed in seen_seed:
                next_seed += 1
            scene = _supplemental_scene_group(
                split=split,
                seed=next_seed,
                task_spec=task_spec,
                template_group=template_group,
            )
            entries.append(
                _entry(
                    template,
                    split=split,
                    seed=next_seed,
                    scene_group_id=scene,
                    source_kind="deterministic_supplemental_held_out_rule",
                )
            )
            supplemental_counts[split] += 1
            seen_seed.add(next_seed)
            next_seed += 1
            cursor += 1
        sealed.extend(entries)
    all_ids = [item["evaluation_id"] for item in [*development, *sealed]]
    if len(all_ids) != len(set(all_ids)):
        raise Phase2CContractError("evaluation schedule contains duplicate identities")
    development_seed_set = {_integer(item["seed"], "development seed") for item in development}
    final_test_seed_set = {
        _integer(item["seed"], "sealed seed")
        for item in sealed
        if item["split"] != "train_distribution_sanity"
    }
    if development_seed_set.intersection(final_test_seed_set):
        raise Phase2CContractError("development and final test seeds overlap")
    semantic = {
        "development": development,
        "sealed": sealed,
        "supplemental_counts": dict(sorted(supplemental_counts.items())),
        "selection_inputs": "Phase 2B split metadata only; no outcomes",
        "candidate_e_paired": True,
        "noop_paired": True,
    }
    return {
        "schema_version": SCHEDULE_SCHEMA,
        **semantic,
        "development_count": len(development),
        "sealed_count": len(sealed),
        "semantic_sha256": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


__all__ = ["SCHEDULE_SCHEMA", "build_phase2c_schedules"]
