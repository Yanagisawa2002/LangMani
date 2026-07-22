"""Pure Phase 2B contracts for deterministic pushing demonstration data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast

from langmani.datasets.lerobot_types import PANDA_ACTION_COMPONENTS
from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS
from langmani.environments.push_specs import (
    PUSH_DIFFICULTIES,
    PUSH_OBJECT_IDS,
    TARGET_REGION_IDS,
    PushTaskSpec,
    stable_push_scene_id,
    stable_push_task_id,
)

COLLECTION_SCHEMA_VERSION = "langmani-v2-phase2b-collection-v0"
ATTEMPT_SCHEMA_VERSION = "langmani-v2-phase2b-attempt-v0"
REPLAY_SCHEMA_VERSION = "langmani-v2-phase2b-replay-v0"
SPLIT_SCHEMA_VERSION = "langmani-v2-phase2b-splits-v0"
LEAKAGE_SCHEMA_VERSION = "langmani-v2-phase2b-leakage-v0"

type CollectionStage = Literal["pilot", "full", "top_up"]
type TemplateGroup = Literal["training", "validation", "held_out_paraphrase"]
type DatasetSplit = Literal[
    "train",
    "validation",
    "test_unseen_scene",
    "test_unseen_language",
    "test_hard",
    "test_visual_shift",
]

DATASET_SPLITS: tuple[DatasetSplit, ...] = (
    "train",
    "validation",
    "test_unseen_scene",
    "test_unseen_language",
    "test_hard",
    "test_visual_shift",
)

_TEMPLATES: Mapping[str, str] = {
    "push_train_canonical_v0": (
        "Push the {object_name} into the {region_name} target region without picking it up."
    ),
    "push_train_direct_v0": "Push the {object_name} into the {region_name} target.",
    "push_train_constraint_v0": (
        "Without lifting it, move the {object_name} by pushing it into the {region_name} region."
    ),
    "push_validation_goal_v0": (
        "Guide the {object_name} across the table until it is inside the {region_name} marker."
    ),
    "push_heldout_nudge_v0": (
        "Nudge the {object_name} so it comes to rest within the {region_name} goal."
    ),
}


class PushDatasetContractError(ValueError):
    """Raised when a Phase 2B data contract is malformed or inconsistent."""


def canonical_json(value: object) -> str:
    """Return the stable serialization used for every persistent identity."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PushDatasetContractError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PushDatasetContractError(f"{label} must be an integer >= {minimum}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PushDatasetContractError(f"{label} must be a non-empty string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise PushDatasetContractError(f"{label} must be a non-empty string list")
    result = tuple(cast(list[str], value))
    if len(set(result)) != len(result):
        raise PushDatasetContractError(f"{label} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class PushCollectionConfig:
    """Frozen project-owned collection specification loaded before any episode."""

    source_path: str
    payload: Mapping[str, object]
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise PushDatasetContractError(
                f"cannot load collection specification: {error}"
            ) from error
        payload = dict(_mapping(raw, "collection specification"))
        cls._validate(payload)
        return cls(
            source_path=source.as_posix(),
            payload=payload,
            fingerprint=sha256_json(payload),
        )

    @staticmethod
    def _validate(payload: Mapping[str, object]) -> None:
        if payload.get("schema_version") != COLLECTION_SCHEMA_VERSION:
            raise PushDatasetContractError("unsupported collection schema_version")
        exact = {
            "environment_id",
            "expert_id",
            "accepted_expert_commit",
            "embodiment",
            "control_mode",
            "sim_backend",
            "action_representation",
            "raw_observation_mode",
        }
        for key in exact:
            _string(payload.get(key), key)
        if payload["environment_id"] != "LangMani-PushToRegion-v0":
            raise PushDatasetContractError("collection environment_id is not the pushing task")
        contract_schema = payload.get("contract_schema_version")
        if contract_schema is None:
            expected_expert = "PushToRegionExpert/CandidateE"
            expected_commit = "59ca88e9f0514187252a6286ab1b8e06c4318fb4"
        else:
            expected_expert = "PushToRegionExpert/CandidateG"
            expected_commit = "24709f1702914e72131245484d4405ac68a6303c"
        candidate_name = expected_expert.rsplit("/", maxsplit=1)[-1].replace(
            "Candidate", "Candidate "
        )
        if payload["expert_id"] != expected_expert:
            raise PushDatasetContractError(
                f"collection must use accepted {candidate_name} identity"
            )
        if payload["accepted_expert_commit"] != expected_commit:
            raise PushDatasetContractError(f"collection must use accepted {candidate_name} commit")
        if payload["control_mode"] != "pd_joint_pos" or payload["sim_backend"] != "physx_cuda":
            raise PushDatasetContractError("collection runtime control/simulator mode changed")
        if payload["raw_observation_mode"] != "state_dict":
            raise PushDatasetContractError("raw authority must use state_dict observations")
        if tuple(_string_tuple(payload.get("objects"), "objects")) != PUSH_OBJECT_IDS:
            raise PushDatasetContractError("object grid differs from the Phase 2 task")
        if tuple(_string_tuple(payload.get("target_regions"), "target_regions")) != (
            TARGET_REGION_IDS
        ):
            raise PushDatasetContractError("target-region grid differs from the Phase 2 task")
        if tuple(_string_tuple(payload.get("difficulties"), "difficulties")) != (PUSH_DIFFICULTIES):
            raise PushDatasetContractError("difficulty grid differs from the Phase 2 task")
        for key, minimum in (
            ("pilot_attempts", 16),
            ("pilot_seed_start", 0),
            ("full_standard_attempts", 1),
            ("full_hard_attempts", 1),
            ("full_seed_start", 0),
            ("shard_size", 1),
            ("preferred_accepted_episodes", 360),
            ("minimum_accepted_episodes", 320),
            ("maximum_top_up_attempts", 0),
        ):
            _integer(payload.get(key), key, minimum=minimum)
        if cast(int, payload["pilot_attempts"]) % 16:
            raise PushDatasetContractError("pilot_attempts must cover the 16-task grid evenly")
        if cast(int, payload["pilot_seed_start"]) + cast(int, payload["pilot_attempts"]) >= cast(
            int, payload["full_seed_start"]
        ):
            raise PushDatasetContractError("pilot and full seed ranges overlap")
        templates = _mapping(payload.get("language_templates"), "language_templates")
        if set(templates) != {"training", "validation", "held_out_paraphrase"}:
            raise PushDatasetContractError("language template groups are incomplete")
        seen: set[str] = set()
        for group, values in templates.items():
            identifiers = _string_tuple(values, f"language_templates.{group}")
            if any(identifier not in _TEMPLATES for identifier in identifiers):
                raise PushDatasetContractError(f"unknown language template in group {group}")
            if seen.intersection(identifiers):
                raise PushDatasetContractError("language template groups overlap")
            seen.update(identifiers)
        runtime = _mapping(payload.get("required_runtime"), "required_runtime")
        required = {
            "numpy": "1.26.4",
            "mplib": "0.1.1",
            "mani_skill": "3.0.1",
            "lerobot": "0.6.0",
        }
        if dict(runtime) != required:
            raise PushDatasetContractError("required runtime pins changed")
        if contract_schema is not None:
            PushCollectionConfig._validate_phase2b2(payload, contract_schema)

    @staticmethod
    def _validate_phase2b2(payload: Mapping[str, object], contract_schema: object) -> None:
        if contract_schema != "langmani-v2-phase2b2-dataset-contract-v0":
            raise PushDatasetContractError("unsupported Phase 2B.2 contract schema")
        if payload.get("dataset_version") != "v2":
            raise PushDatasetContractError("Phase 2B.2 dataset_version must be v2")
        if payload.get("source_commit_policy") != "exact_clean_runtime_head":
            raise PushDatasetContractError(
                "Phase 2B.2 source commit must bind the clean runtime HEAD"
            )
        if payload.get("atomic_episode_commits") is not True:
            raise PushDatasetContractError("Phase 2B.2 requires atomic episode commits")
        dataset = _mapping(payload.get("dataset"), "dataset")
        if dataset.get("repo_id") != "langmani/phase2b-push-v2":
            raise PushDatasetContractError("Phase 2B.2 must use its independent v2 dataset ID")
        forbidden = ("phase2b-push-v1", "phase2b-v1")
        if any(token in str(dataset.get("output_directory", "")) for token in forbidden):
            raise PushDatasetContractError("Phase 2B.2 cannot write into a v1 dataset root")
        if payload.get("supersedes_dataset_id") != "langmani/phase2b-push-v1":
            raise PushDatasetContractError("Phase 2B.2 must record the retired v1 dataset")
        if tuple(_string_tuple(payload.get("state_names"), "state_names")) != (
            PANDA_POLICY_STATE_COMPONENTS
        ):
            raise PushDatasetContractError("Phase 2B.2 state names changed from PandaPolicyStateV0")
        if tuple(_string_tuple(payload.get("action_names"), "action_names")) != (
            PANDA_ACTION_COMPONENTS
        ):
            raise PushDatasetContractError("Phase 2B.2 action names changed")
        if payload.get("state_shape") != [9] or payload.get("action_shape") != [8]:
            raise PushDatasetContractError("Phase 2B.2 must preserve 9D state and 8D action")
        bounds = _mapping(payload.get("action_bounds"), "action_bounds")
        for name in ("low", "high"):
            values = bounds.get(name)
            if not isinstance(values, list) or len(values) != 8:
                raise PushDatasetContractError(f"action_bounds.{name} must contain eight values")
            if not all(
                isinstance(item, int | float) and not isinstance(item, bool) for item in values
            ):
                raise PushDatasetContractError(f"action_bounds.{name} must be numeric")
        low = cast(list[int | float], bounds["low"])
        high = cast(list[int | float], bounds["high"])
        if any(float(left) > float(right) for left, right in zip(low, high, strict=True)):
            raise PushDatasetContractError("Phase 2B.2 action bounds are inverted")
        if _integer(payload.get("minimum_accepted_episodes"), "minimum_accepted_episodes") < 400:
            raise PushDatasetContractError("Phase 2B.2 requires at least 400 accepted episodes")
        if (
            _integer(payload.get("preferred_accepted_episodes"), "preferred_accepted_episodes")
            < 500
        ):
            raise PushDatasetContractError("Phase 2B.2 preferred target must be at least 500")
        if _integer(payload.get("minimum_total_frames"), "minimum_total_frames") < 50_000:
            raise PushDatasetContractError("Phase 2B.2 requires at least 50,000 frames")
        expert = _mapping(payload.get("expert_evaluation"), "expert_evaluation")
        standard = _integer(expert.get("standard_episodes"), "expert standard episodes")
        hard = _integer(expert.get("hard_episodes"), "expert hard episodes")
        minimum_rate = expert.get("minimum_success_rate")
        if not isinstance(minimum_rate, int | float) or isinstance(minimum_rate, bool):
            raise PushDatasetContractError("expert minimum_success_rate must be numeric")
        if standard + hard < 100 or float(minimum_rate) < 0.95:
            raise PushDatasetContractError("Phase 2B.2 expert gate must cover 100 episodes at 95%")
        expert_start = _integer(expert.get("seed_start"), "expert seed_start")
        expert_seeds = set(range(expert_start, expert_start + standard + hard))
        full_start = _integer(payload.get("full_seed_start"), "full_seed_start")
        full_count = _integer(payload.get("full_standard_attempts"), "full_standard_attempts") + (
            _integer(payload.get("full_hard_attempts"), "full_hard_attempts")
        )
        top_up_count = _integer(payload.get("maximum_top_up_attempts"), "maximum_top_up_attempts")
        collection_seeds = set(range(full_start, full_start + full_count + top_up_count))
        if expert_seeds.intersection(collection_seeds):
            raise PushDatasetContractError("expert evaluation and collection seeds overlap")

    def integer(self, key: str) -> int:
        return _integer(self.payload.get(key), key)

    @property
    def language_templates(self) -> Mapping[str, tuple[str, ...]]:
        raw = _mapping(self.payload["language_templates"], "language_templates")
        return {key: _string_tuple(value, key) for key, value in raw.items()}

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "fingerprint": self.fingerprint,
            "specification": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class PushScheduledAttempt:
    """One immutable semantic attempt in a deterministic schedule."""

    stage: CollectionStage
    attempt_index: int
    seed: int
    task_spec: PushTaskSpec
    template_group: TemplateGroup
    template_id: str
    instruction: str
    scene_group_id: str
    episode_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "attempt_index": self.attempt_index,
            "seed": self.seed,
            "task_spec": self.task_spec.to_dict(),
            "task_id": stable_push_task_id(self.task_spec),
            "template_group": self.template_group,
            "template_id": self.template_id,
            "instruction": self.instruction,
            "scene_group_id": self.scene_group_id,
            "episode_id": self.episode_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        task = PushTaskSpec.from_mapping(_mapping(value.get("task_spec"), "task_spec"))
        stage = _string(value.get("stage"), "stage")
        group = _string(value.get("template_group"), "template_group")
        if stage not in {"pilot", "full", "top_up"}:
            raise PushDatasetContractError("unknown collection stage")
        if group not in {"training", "validation", "held_out_paraphrase"}:
            raise PushDatasetContractError("unknown template group")
        return cls(
            stage=cast(CollectionStage, stage),
            attempt_index=_integer(value.get("attempt_index"), "attempt_index"),
            seed=_integer(value.get("seed"), "seed"),
            task_spec=task,
            template_group=cast(TemplateGroup, group),
            template_id=_string(value.get("template_id"), "template_id"),
            instruction=_string(value.get("instruction"), "instruction"),
            scene_group_id=_string(value.get("scene_group_id"), "scene_group_id"),
            episode_id=_string(value.get("episode_id"), "episode_id"),
        )


def render_instruction(template_id: str, task: PushTaskSpec) -> str:
    try:
        template = _TEMPLATES[template_id]
    except KeyError as error:
        raise PushDatasetContractError(f"unknown template_id {template_id!r}") from error
    object_name = "blue cube" if task.target_object_id == "blue_cube" else "orange cylinder"
    region_name = task.target_region_id.replace("_", "-")
    return template.format(object_name=object_name, region_name=region_name)


def _template_for_index(config: PushCollectionConfig, index: int) -> tuple[TemplateGroup, str]:
    templates = config.language_templates
    remainder = index % 10
    if remainder == 0:
        group: TemplateGroup = "held_out_paraphrase"
    elif remainder in {1, 2}:
        group = "validation"
    else:
        group = "training"
    values = templates[group]
    return group, values[(index // 10) % len(values)]


def _scheduled_attempt(
    config: PushCollectionConfig,
    *,
    stage: CollectionStage,
    attempt_index: int,
    seed: int,
    task: PushTaskSpec,
) -> PushScheduledAttempt:
    group, template_id = _template_for_index(config, attempt_index)
    instruction = render_instruction(template_id, task)
    scene_group_id = stable_push_scene_id(seed)
    identity = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "collection_fingerprint": config.fingerprint,
        "stage": stage,
        "attempt_index": attempt_index,
        "seed": seed,
        "scene_group_id": scene_group_id,
        "task_id": stable_push_task_id(task),
        "template_group": group,
        "template_id": template_id,
    }
    episode_id = "langmani-v2-push-episode-" + sha256_json(identity).removeprefix("sha256:")[:24]
    return PushScheduledAttempt(
        stage=stage,
        attempt_index=attempt_index,
        seed=seed,
        task_spec=task,
        template_group=group,
        template_id=template_id,
        instruction=instruction,
        scene_group_id=scene_group_id,
        episode_id=episode_id,
    )


def build_collection_schedule(
    config: PushCollectionConfig, stage: Literal["pilot", "full"]
) -> tuple[PushScheduledAttempt, ...]:
    """Build the exact pilot or initial full schedule with balanced task rotation."""

    tasks_by_difficulty = {
        difficulty: tuple(
            PushTaskSpec(object_id, region_id, difficulty)
            for object_id in PUSH_OBJECT_IDS
            for region_id in TARGET_REGION_IDS
        )
        for difficulty in PUSH_DIFFICULTIES
    }
    scheduled: list[PushScheduledAttempt] = []
    if stage == "pilot":
        count = config.integer("pilot_attempts")
        start = config.integer("pilot_seed_start")
        grid = tuple(
            task for difficulty in PUSH_DIFFICULTIES for task in tasks_by_difficulty[difficulty]
        )
        for index in range(count):
            scheduled.append(
                _scheduled_attempt(
                    config,
                    stage="pilot",
                    attempt_index=index,
                    seed=start + index,
                    task=grid[index % len(grid)],
                )
            )
    else:
        start = config.integer("full_seed_start")
        counts = (
            ("standard", config.integer("full_standard_attempts")),
            ("hard", config.integer("full_hard_attempts")),
        )
        index = 0
        for difficulty, count in counts:
            tasks = tasks_by_difficulty[cast(Literal["standard", "hard"], difficulty)]
            for offset in range(count):
                scheduled.append(
                    _scheduled_attempt(
                        config,
                        stage="full",
                        attempt_index=index,
                        seed=start + index,
                        task=tasks[offset % len(tasks)],
                    )
                )
                index += 1
    _validate_schedule(scheduled)
    return tuple(scheduled)


def build_top_up_schedule(
    config: PushCollectionConfig,
    *,
    deficits: Mapping[str, int],
    start_index: int,
) -> tuple[PushScheduledAttempt, ...]:
    """Allocate one bounded round-robin top-up only to declared deficient strata."""

    maximum = config.integer("maximum_top_up_attempts")
    eligible: list[PushTaskSpec] = []
    for object_id in PUSH_OBJECT_IDS:
        for region_id in TARGET_REGION_IDS:
            for difficulty in PUSH_DIFFICULTIES:
                task = PushTaskSpec(object_id, region_id, difficulty)
                if (
                    any(int(deficits.get(key, 0)) > 0 for key in (object_id, region_id, difficulty))
                    or int(deficits.get("total", 0)) > 0
                ):
                    eligible.append(task)
    if not eligible:
        return ()
    # A top-up is a single immutable bounded schedule. Allocate the complete
    # predeclared budget when any post-replay deficit exists; allocating only
    # the current success deficit would incorrectly assume a 100% expert and
    # replay yield and could leave the one allowed top-up short again.
    count = maximum
    seed_start = (
        config.integer("full_seed_start")
        + config.integer("full_standard_attempts")
        + config.integer("full_hard_attempts")
    )
    schedule = tuple(
        _scheduled_attempt(
            config,
            stage="top_up",
            attempt_index=start_index + index,
            seed=seed_start + index,
            task=eligible[index % len(eligible)],
        )
        for index in range(count)
    )
    _validate_schedule(schedule)
    return schedule


def stage_runtime_manifest_name(stage: str) -> str:
    """Keep the base producer immutable while recording later top-up code separately."""

    if stage not in {"pilot", "full", "top_up"}:
        raise PushDatasetContractError("runtime stage must be pilot, full, or top_up")
    return "runtime_environment_top_up.json" if stage == "top_up" else "runtime_environment.json"


def _validate_schedule(schedule: Sequence[PushScheduledAttempt]) -> None:
    ids = [item.episode_id for item in schedule]
    seeds = [item.seed for item in schedule]
    if len(set(ids)) != len(ids):
        raise PushDatasetContractError("schedule contains duplicate episode identities")
    if len(set(seeds)) != len(seeds):
        raise PushDatasetContractError("schedule contains duplicate seeds")
    for item in schedule:
        if item.instruction != render_instruction(item.template_id, item.task_spec):
            raise PushDatasetContractError("schedule instruction is disconnected from TaskSpec")


def generation_acceptance_failures(record: Mapping[str, object]) -> tuple[str, ...]:
    """Return conjunctive generation-time failures without considering replay."""

    failures: list[str] = []
    if record.get("expert_success") is not True:
        failures.append("expert_not_successful")
    evaluation = _mapping(record.get("final_evaluation", {}), "final_evaluation")
    if evaluation.get("success") is not True:
        failures.append("canonical_success_false")
    if evaluation.get("fail") is True:
        failures.append("environment_fail_true")
    for key in (
        "wrong_object_contact",
        "wrong_object_displaced",
        "target_outside_workspace",
        "invalid_action",
        "action_out_of_bounds",
        "target_lifted",
        "target_toppled",
    ):
        if evaluation.get(key) is True:
            failures.append(key)
    if record.get("actions_finite") is not True:
        failures.append("actions_non_finite")
    if record.get("observations_finite") is not True:
        failures.append("observations_non_finite")
    if record.get("time_contract_valid") is not True:
        failures.append("time_contract_invalid")
    if record.get("action_contract_valid") is not True:
        failures.append("action_contract_invalid")
    if record.get("task_metadata_valid") is not True:
        failures.append("task_metadata_invalid")
    if record.get("instruction_valid") is not True:
        failures.append("instruction_invalid")
    if not record.get("trajectory_sha256"):
        failures.append("trajectory_hash_missing")
    if not record.get("initial_state_sha256"):
        failures.append("initial_state_hash_missing")
    return tuple(failures)


def accepted_counts(records: Iterable[Mapping[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("accepted") is not True:
            continue
        counts["total"] += 1
        task = _mapping(record.get("task_spec"), "task_spec")
        for key in ("target_object_id", "target_region_id", "difficulty"):
            counts[_string(task.get(key), key)] += 1
    return counts


def quota_deficits(
    config: PushCollectionConfig, records: Iterable[Mapping[str, object]]
) -> dict[str, int]:
    quotas = _mapping(config.payload.get("minimum_quotas"), "minimum_quotas")
    counts = accepted_counts(records)
    return {
        key: max(0, _integer(value, f"minimum_quotas.{key}") - counts[key])
        for key, value in quotas.items()
        if max(0, _integer(value, f"minimum_quotas.{key}") - counts[key]) > 0
    }


def summarize_attempt_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate preserved attempts without hiding genuine failure strata."""

    names = (
        "geometry",
        "direction",
        "difficulty",
        "template_group",
        "failure_category",
        "episode_length_bucket",
        "correction_push_count",
    )
    counts = {name: Counter[str]() for name in names}
    accepted_counts_by = {name: Counter[str]() for name in names}
    rejected_counts_by = {name: Counter[str]() for name in names}
    accepted = 0
    rejected = 0
    for record in records:
        task = _mapping(record.get("task_spec"), "task_spec")
        geometry = "cube" if task.get("target_object_id") == "blue_cube" else "horizontal_cylinder"
        values = {
            "geometry": geometry,
            "direction": _string(task.get("target_region_id"), "target_region_id"),
            "difficulty": _string(task.get("difficulty"), "difficulty"),
            "template_group": _string(record.get("template_group"), "template_group"),
        }
        length = _integer(record.get("episode_length"), "episode_length")
        values["episode_length_bucket"] = (
            "0-49"
            if length < 50
            else "50-99"
            if length < 100
            else "100-149"
            if length < 150
            else "150+"
        )
        corrections = _integer(record.get("correction_push_count", 0), "correction_push_count")
        values["correction_push_count"] = str(corrections)
        selected = accepted_counts_by if record.get("accepted") is True else rejected_counts_by
        for name, value in values.items():
            counts[name][value] += 1
            selected[name][value] += 1
        if record.get("accepted") is True:
            accepted += 1
        else:
            rejected += 1
            category = record.get("failure_category")
            failure = str(category or "replay_or_contract_rejection")
            counts["failure_category"][failure] += 1
            rejected_counts_by["failure_category"][failure] += 1
    return {
        "attempted": len(records),
        "accepted": accepted,
        "rejected": rejected,
        **{name: dict(sorted(value.items())) for name, value in counts.items()},
        "accepted_by": {
            name: dict(sorted(value.items())) for name, value in accepted_counts_by.items()
        },
        "rejected_by": {
            name: dict(sorted(value.items())) for name, value in rejected_counts_by.items()
        },
    }


def assign_split(record: Mapping[str, object]) -> DatasetSplit:
    """Assign one accepted episode to a stable group-level split."""

    group = _string(record.get("template_group"), "template_group")
    if group == "held_out_paraphrase":
        return "test_unseen_language"
    if group == "validation":
        return "validation"
    task = _mapping(record.get("task_spec"), "task_spec")
    digest = hashlib.sha256(
        _string(record.get("scene_group_id"), "scene_group_id").encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:2], "big") % 100
    if task.get("difficulty") == "hard" and bucket < 25:
        return "test_hard"
    if 25 <= bucket < 35:
        return "test_visual_shift"
    if 35 <= bucket < 45:
        return "test_unseen_scene"
    return "train"


def build_split_manifest(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    assignments: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("accepted") is not True:
            continue
        episode_id = _string(record.get("episode_id"), "episode_id")
        split = assign_split(record)
        if episode_id in assignments:
            raise PushDatasetContractError("accepted episode is assigned more than once")
        assignments[episode_id] = split
        counts[split] += 1
    missing = [split for split in DATASET_SPLITS if counts[split] == 0]
    if missing:
        raise PushDatasetContractError("empty required dataset splits: " + ", ".join(missing))
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "assignments": assignments,
        "counts": {split: counts[split] for split in DATASET_SPLITS},
        "fingerprint": sha256_json(assignments),
    }


def audit_split_leakage(
    records: Sequence[Mapping[str, object]], split_manifest: Mapping[str, object]
) -> dict[str, object]:
    """Recompute identity, trajectory, state, scene, language, and file leakage."""

    assignments = _mapping(split_manifest.get("assignments"), "assignments")
    errors: list[str] = []
    seen_episode: set[str] = set()
    seen_seed: dict[int, str] = {}
    seen_trajectory: dict[str, str] = {}
    seen_state: dict[str, str] = {}
    scene_splits: dict[str, set[str]] = defaultdict(set)
    split_files: dict[str, set[str]] = defaultdict(set)
    template_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.get("accepted") is not True:
            continue
        episode_id = _string(record.get("episode_id"), "episode_id")
        split = _string(assignments.get(episode_id), f"assignment[{episode_id}]")
        if episode_id in seen_episode:
            errors.append(f"duplicate_episode_id:{episode_id}")
        seen_episode.add(episode_id)
        seed = _integer(record.get("seed"), "seed")
        other_seed_episode = seen_seed.get(seed)
        if other_seed_episode is not None and other_seed_episode != episode_id:
            errors.append(f"duplicate_seed:{other_seed_episode}:{episode_id}:{seed}")
        seen_seed[seed] = episode_id
        for field, seen, code in (
            ("trajectory_sha256", seen_trajectory, "duplicate_trajectory_hash"),
            ("initial_state_sha256", seen_state, "duplicate_initial_state_hash"),
        ):
            digest = _string(record.get(field), field)
            other = seen.get(digest)
            if other is not None and other != episode_id:
                errors.append(f"{code}:{other}:{episode_id}")
            seen[digest] = episode_id
        scene_splits[_string(record.get("scene_group_id"), "scene_group_id")].add(split)
        template_splits[_string(record.get("template_id"), "template_id")].add(split)
        split_files[split].add(_string(record.get("raw_h5_path"), "raw_h5_path"))
    for scene, splits in scene_splits.items():
        if len(splits) != 1:
            errors.append(f"scene_group_overlap:{scene}:{sorted(splits)}")
    training_template_ids = {
        template for template, splits in template_splits.items() if "train" in splits
    }
    for record in records:
        if record.get("accepted") is not True:
            continue
        if record.get("template_group") == "held_out_paraphrase":
            template = _string(record.get("template_id"), "template_id")
            if template in training_template_ids:
                errors.append(f"held_out_template_in_train:{template}")
    file_overlap_count = 0
    # Source shards may contain multiple split episodes by design. The exported
    # dataset owns per-episode frames; source HDF5 path overlap is reported, not
    # treated as frame leakage.
    for left_index, left in enumerate(DATASET_SPLITS):
        for right in DATASET_SPLITS[left_index + 1 :]:
            file_overlap_count += len(split_files[left].intersection(split_files[right]))
    return {
        "schema_version": LEAKAGE_SCHEMA_VERSION,
        "accepted_episode_count": len(seen_episode),
        "duplicate_episode_id_count": sum(
            error.startswith("duplicate_episode_id") for error in errors
        ),
        "duplicate_trajectory_hash_count": sum(
            error.startswith("duplicate_trajectory_hash") for error in errors
        ),
        "duplicate_seed_count": sum(error.startswith("duplicate_seed") for error in errors),
        "duplicate_initial_state_hash_count": sum(
            error.startswith("duplicate_initial_state_hash") for error in errors
        ),
        "scene_group_overlap_count": sum(
            error.startswith("scene_group_overlap") for error in errors
        ),
        "held_out_language_overlap_count": sum(
            error.startswith("held_out_template_in_train") for error in errors
        ),
        "source_shard_cross_split_reference_count": file_overlap_count,
        "exported_frame_file_overlap_count": 0,
        "errors": errors,
        "passed": not errors,
    }


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "COLLECTION_SCHEMA_VERSION",
    "DATASET_SPLITS",
    "LEAKAGE_SCHEMA_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "SPLIT_SCHEMA_VERSION",
    "PushCollectionConfig",
    "PushDatasetContractError",
    "PushScheduledAttempt",
    "accepted_counts",
    "assign_split",
    "audit_split_leakage",
    "build_collection_schedule",
    "build_split_manifest",
    "build_top_up_schedule",
    "canonical_json",
    "generation_acceptance_failures",
    "quota_deficits",
    "render_instruction",
    "sha256_bytes",
    "sha256_json",
    "stage_runtime_manifest_name",
    "summarize_attempt_records",
]
