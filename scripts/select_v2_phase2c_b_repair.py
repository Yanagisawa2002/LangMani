"""Bind the sole Phase 2C-B bounded repair to demonstrated horizon evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c import sha256_file
from langmani.v2.phase2c_a import EXECUTION_HORIZONS, PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import Phase2CBContractError, canonical_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-policy", type=Path, required=True)
    parser.add_argument("--initial-validation-summary", type=Path, required=True)
    parser.add_argument("--repair-screen-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_fingerprinted(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CBContractError(f"cannot read repair evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CBContractError(f"{path} must contain one object")
    semantic = dict(value)
    fingerprint = semantic.pop("fingerprint", None)
    if fingerprint not in {
        canonical_fingerprint(semantic),
        f"sha256:{sha256_hex(semantic)}",
    }:
        raise Phase2CBContractError(f"repair evidence fingerprint changed: {path}")
    return value


def _integer(document: Mapping[str, object], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase2CBContractError(f"repair evidence lacks valid integer {key}")
    return value


def _failure_count(document: Mapping[str, object], category: str) -> int:
    categories = document.get("failure_categories")
    value = categories.get(category, 0) if isinstance(categories, Mapping) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase2CBContractError(f"repair evidence has invalid failure category {category}")
    return value


def _path_record(path: Path, document: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": f"sha256:{sha256_file(path)}",
        "fingerprint": document["fingerprint"],
    }


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CBContractError(f"existing immutable repair differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    paths = {
        "selected_policy": args.selected_policy.resolve(),
        "initial_validation_summary": args.initial_validation_summary.resolve(),
        "repair_screen_summary": args.repair_screen_summary.resolve(),
    }
    documents = {key: _read_fingerprinted(path) for key, path in paths.items()}
    selected = documents["selected_policy"]
    initial = documents["initial_validation_summary"]
    screen = documents["repair_screen_summary"]

    selected_checkpoint = selected.get("selected_checkpoint_sha256")
    original_horizon = selected.get("selected_execution_horizon")
    initial_identities = initial.get("checkpoint_identities")
    screen_identities = screen.get("checkpoint_identities")
    repaired_horizon = screen.get("execution_horizon")
    if (
        selected.get("schema_version") != "langmani-v2-phase2c-b-selected-policy-v0"
        or selected.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or selected.get("model_kind") != "pick_smolvla"
        or selected.get("task_id") != "PickCube-v1"
        or selected.get("selection_split") != "validation"
        or selected.get("final_test_opened") is not False
        or selected.get("passed") is not True
        or not isinstance(selected_checkpoint, str)
        or not isinstance(original_horizon, int)
        or isinstance(original_horizon, bool)
        or original_horizon not in EXECUTION_HORIZONS
    ):
        raise Phase2CBContractError("original selected policy is outside bounded repair scope")
    common_summary_contract = (
        ("package_fingerprint", PACKAGE_FINGERPRINT),
        ("model_kind", "pick_smolvla"),
        ("task_id", "PickCube-v1"),
        ("split", "validation"),
        ("instruction_condition", "correct"),
        ("completed", True),
    )
    if any(initial.get(key) != value for key, value in common_summary_contract) or (
        _integer(initial, "episode_count") != 30
        or _integer(initial, "success_count") != 0
        or _integer(initial, "invalid_action_count") != 0
        or _integer(initial, "simulator_error_count") != 0
        or initial_identities != [selected_checkpoint]
        or initial.get("execution_horizon") != original_horizon
    ):
        raise Phase2CBContractError("initial 0/30 Pick gate does not permit repair")
    if any(screen.get(key) != value for key, value in common_summary_contract) or (
        _integer(screen, "episode_count") != selected.get("selection_subset_episode_count")
        or _integer(screen, "invalid_action_count") != 0
        or _integer(screen, "simulator_error_count") != 0
        or screen_identities != [selected_checkpoint]
        or not isinstance(repaired_horizon, int)
        or isinstance(repaired_horizon, bool)
        or repaired_horizon not in EXECUTION_HORIZONS
        or repaired_horizon == original_horizon
    ):
        raise Phase2CBContractError("repair screen is outside the locked horizon grid")

    candidates = selected.get("candidates")
    if not isinstance(candidates, list) or not any(
        isinstance(candidate, Mapping)
        and candidate.get("checkpoint_sha256") == selected_checkpoint
        and candidate.get("execution_horizon") == repaired_horizon
        and candidate.get("summary_fingerprint") == screen.get("fingerprint")
        for candidate in candidates
    ):
        raise Phase2CBContractError("repair screen was not part of the frozen policy grid")

    initial_episodes = _integer(initial, "episode_count")
    screen_episodes = _integer(screen, "episode_count")
    initial_no_motion = _failure_count(initial, "no_initial_motion")
    screen_no_motion = _failure_count(screen, "no_initial_motion")
    initial_motion_rate = (initial_episodes - initial_no_motion) / initial_episodes
    screen_motion_rate = (screen_episodes - screen_no_motion) / screen_episodes
    post_contact = screen.get("post_contact_divergence_mean_m")
    if (
        screen_motion_rate <= initial_motion_rate
        or not isinstance(post_contact, int | float)
        or not math.isfinite(float(post_contact))
        or float(post_contact) <= 0.0
    ):
        raise Phase2CBContractError(
            "repair screen does not demonstrate a coherent-motion horizon defect"
        )

    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-bounded-repair-policy-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "repair_ordinal": 1,
        "repair_kind": "action_chunk_execution_horizon",
        "demonstrated_defect": (
            "the originally selected horizon suppressed coherent initial motion while "
            "a pre-locked alternate horizon produced contact-bearing behavior"
        ),
        "inputs": {key: _path_record(paths[key], documents[key]) for key in sorted(paths)},
        "selected_checkpoint_sha256": selected_checkpoint,
        "original_execution_horizon": original_horizon,
        "repaired_execution_horizon": repaired_horizon,
        "initial_episode_count": initial_episodes,
        "initial_no_initial_motion_count": initial_no_motion,
        "initial_motion_episode_rate": initial_motion_rate,
        "repair_screen_episode_count": screen_episodes,
        "repair_screen_no_initial_motion_count": screen_no_motion,
        "repair_screen_motion_episode_rate": screen_motion_rate,
        "repair_screen_post_contact_divergence_mean_m": float(post_contact),
        "model_bytes_changed": False,
        "optimizer_created": False,
        "training_started": False,
        "dataset_changed": False,
        "split_changed": False,
        "observation_contract_changed": False,
        "physical_action_contract_changed": False,
        "success_predicate_changed": False,
        "final_test_opened": False,
        "passed": True,
    }
    result = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
    _write_new_or_equal(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
