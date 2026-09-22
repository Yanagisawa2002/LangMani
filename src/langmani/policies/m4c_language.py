"""Auditable language-only views over unchanged M4B robot trajectories."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from langmani.policies.m4a_data import digest
from langmani.policies.m4b_protocol import GOALS, prompts

LEVELS = (1, 5, 10)
FAMILIES = ("lexical", "syntactic", "referential", "polite_prefix", "natural")
TRAIN = (
    ("canonical", "Pick up the red cube and place it in the {side} bin."),
    ("lexical", "Lift the red block and set it in the {side} container."),
    ("syntactic", "After picking up the red cube, place it in the bin on the {side}."),
    ("referential", "Pick up the red cube and place that cube in the {side}-hand bin."),
    ("polite_prefix", "Please pick up the red cube and place it in the {side} bin."),
    ("natural", "The red cube should end up in the {side} bin. Pick it up and move it there."),
    ("lexical", "Take hold of the red cube and deposit it into the {side} bin."),
    ("syntactic", "Put the red cube into the bin on the {side} after lifting it."),
    ("referential", "Pick up the cube colored red, then put it in the {side} bin."),
    ("polite_prefix", "For this task, pick up the red cube and place it in the {side} bin."),
)
HELD_OUT = {
    "lexical": (
        "Pick up the scarlet cube and place it in the {side} receptacle.",
        "Raise the red block and put it in the {side} tray.",
    ),
    "syntactic": (
        "It is the red cube that you must pick up and place in the bin on the {side}.",
        "The bin on the {side} is where you should place the red cube after picking it up.",
    ),
    "natural": (
        "Can you get the red cube off the table and into the {side} bin?",
        "Move the red cube from its spot on the table into the bin to the {side}.",
    ),
}


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def side(goal: str) -> str:
    if goal not in GOALS:
        raise ValueError("unsupported semantic goal")
    return goal.split(":")[1].split("_")[0]


def expression(template_id: str, family: str, template: str, goal: str) -> dict[str, str]:
    if template.count("{side}") != 1:
        raise ValueError("one explicit side placeholder is required")
    return {
        "template_id": template_id,
        "paraphrase_family": family,
        "semantic_goal_id": goal,
        "instruction_text": template.format(side=side(goal)),
    }


def catalogs() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = [
        expression(f"train_{i:02d}", family, template, goal)
        for i, (family, template) in enumerate(TRAIN)
        for goal in GOALS
    ]
    held = [
        expression(f"held_{family}_{i:02d}", family, template, goal)
        for family, templates in HELD_OUT.items()
        for i, template in enumerate(templates)
        for goal in GOALS
    ]
    return train, held


def validate_catalogs(train: list[dict[str, str]], held: list[dict[str, str]]) -> dict[str, Any]:
    seen_text: set[str] = set()
    seen_ids: set[tuple[str, str]] = set()
    for row in train + held:
        text = normalize(row["instruction_text"])
        goal = row["semantic_goal_id"]
        expected_side = side(goal)
        other_side = "right" if expected_side == "left" else "left"
        words = text.split()
        if not text or expected_side not in words or other_side in words:
            raise ValueError("instruction has missing/contradictory side semantics")
        if not ({"red", "scarlet"} & set(words)) or not ({"cube", "block"} & set(words)):
            raise ValueError("instruction must explicitly identify the red object")
        if text in seen_text:
            raise ValueError("normalized instruction overlap or duplicate")
        seen_text.add(text)
        key = row["template_id"], goal
        if key in seen_ids:
            raise ValueError("duplicate template ID within a goal")
        seen_ids.add(key)
    train_ids = {r["template_id"] for r in train}
    held_ids = {r["template_id"] for r in held}
    if train_ids & held_ids:
        raise ValueError("forbidden train/held-out template ID overlap")
    forbidden = {
        normalize(p["instruction"])
        for i in (0, 1)
        for p in prompts(i)
        if p["prompt_id"].startswith("paraphrase")
    }
    if forbidden & {normalize(r["instruction_text"]) for r in train}:
        raise ValueError("original M4B held-out paraphrase reused in training")
    if not set(FAMILIES) <= {r["paraphrase_family"] for r in train}:
        raise ValueError("training catalog must cover all five controlled families")
    return {
        "passed": True,
        "training_expressions": len(train),
        "held_out_expressions": len(held),
        "forbidden_m4b_expressions": len(forbidden),
        "normalized_overlap": 0,
        "forbidden_template_id_overlap": 0,
        "normalization": "Unicode NFKC, casefold, punctuation-to-space, whitespace collapse",
    }


def make_manifest(split: dict[str, Any]) -> dict[str, Any]:
    train, held = catalogs()
    validation = validate_catalogs(train, held)
    episodes = [r for r in split["episodes"] if r["episode_id"] in split["train_episode_ids"]]
    if (
        len(split["train_episode_ids"]) != len(set(split["train_episode_ids"]))
        or len(episodes) != len(set(split["train_episode_ids"]))
        or any(r["split"] != "train" for r in episodes)
        or sum(r["frames"] for r in episodes) != split["train_frames"]
    ):
        raise ValueError("source training episode identities/frames differ")
    views = {}
    for diversity in LEVELS:
        templates = train[: diversity * 2]
        variants = [
            {
                **text,
                "episode_id": ep["episode_id"],
                "scene_seed": ep["scene_seed"],
                "scene_group_id": ep["scene_group_id"],
                "trajectory_source": ep["raw_episode_id"],
                "frames": ep["frames"],
            }
            for ep in episodes
            for text in templates
            if text["semantic_goal_id"] == ep["semantic_goal"]
        ]
        if len(variants) != diversity * len(episodes):
            raise ValueError("missing semantic-goal realization")
        payload = {
            "diversity_per_goal": diversity,
            "unique_robot_trajectories": len(episodes),
            "unique_robot_frames": split["train_frames"],
            "label_realizations": len(variants),
            "templates": templates,
            "episode_realizations": variants,
        }
        views[f"L{diversity}"] = {**payload, "view_sha256": digest(payload)}
    payload = {
        "schema_version": "langmani-m4c-language-v1",
        "source_split_sha256": split["split_sha256"],
        "source_content_sha256": split["content_sha256"],
        "language_seed": 0,
        "training_catalog": train,
        "held_out_catalog": held,
        "views": views,
        "validation": validation,
        "sampling": "SHA256(seed, ordinal, episode, frame) modulo diversity; original frame sampler",
    }
    return {**payload, "language_manifest_sha256": digest(payload)}


def check_manifest(manifest: dict[str, Any], split: dict[str, Any]) -> None:
    validate_catalogs(manifest["training_catalog"], manifest["held_out_catalog"])
    if manifest != make_manifest(split):
        raise ValueError("language manifest differs from committed catalog/source split")


def sample_expression(
    manifest: dict[str, Any], level: str, goal: str, ordinal: int, episode: int, frame: int
) -> dict[str, str]:
    if ordinal < 0 or episode < 0 or frame < 0:
        raise ValueError("sample identity must be nonnegative")
    templates = [r for r in manifest["views"][level]["templates"] if r["semantic_goal_id"] == goal]
    if len(templates) != manifest["views"][level]["diversity_per_goal"]:
        raise ValueError("language view lacks the requested goal")
    key = f"{manifest['language_seed']}:{ordinal}:{episode}:{frame}".encode()
    choice = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(templates)
    return templates[choice]


def make_schedule(manifest: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    scenes = []
    for i, scene in enumerate(source["scenes"]):
        prompt_rows = []
        for kind in ("seen", "lexical", "syntactic", "natural"):
            for goal in GOALS:
                catalog = (
                    manifest["training_catalog"] if kind == "seen" else manifest["held_out_catalog"]
                )
                template_id = "train_00" if kind == "seen" else f"held_{kind}_{i % 2:02d}"
                text = next(
                    r
                    for r in catalog
                    if r["template_id"] == template_id and r["semantic_goal_id"] == goal
                )
                prompt_rows.append(
                    {
                        "prompt_id": f"{kind}_{side(goal)}",
                        "condition": kind,
                        "prompt_goal": goal,
                        "instruction": text["instruction_text"],
                        "template_id": template_id,
                        "paraphrase_family": text["paraphrase_family"],
                    }
                )
        prompt_rows.append(
            {
                "prompt_id": "blank",
                "condition": "blank",
                "prompt_goal": None,
                "instruction": "",
                "template_id": None,
                "paraphrase_family": None,
            }
        )
        scenes.append({"seed": scene["seed"], "prompts": prompt_rows})
    payload = {
        "schema_version": "langmani-m4c-schedule-v1",
        "source_m4b_schedule_sha256": source["schedule_sha256"],
        "language_manifest_sha256": manifest["language_manifest_sha256"],
        "scenes": scenes,
        "physical_rollouts": len(scenes) * 9,
        "scored_rows": len(scenes) * 12,
        "max_episode_steps": 200,
        "seen_scope": "canonical shared by every training group",
    }
    return {**payload, "schedule_sha256": digest(payload)}
