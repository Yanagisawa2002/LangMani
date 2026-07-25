"""Compare correct, wrong, and shuffled shared-ACT Task-ID interventions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT, TASK_IDS


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise RuntimeError(f"{path} is empty")
    return rows


def _actions(record: dict[str, object]) -> np.ndarray:
    value = record.get("action_sequence")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1:] != (8,) or not np.isfinite(array).all():
        raise RuntimeError("intervention episode lacks a finite float32[N,8] action trace")
    fingerprint = f"sha256:{sha256_hex(tuple(tuple(float(x) for x in row) for row in array))}"
    if record.get("action_sequence_fingerprint") != fingerprint:
        raise RuntimeError("intervention action-sequence fingerprint changed")
    return array


def _optional_float(value: object) -> float:
    if value is None:
        return 0.0
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise RuntimeError("task-intervention motion diagnostic is invalid")
    return float(value)


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise RuntimeError("task-intervention aggregate count is invalid")
    return value


def _comparison(
    correct: dict[str, object],
    intervention: dict[str, object],
    *,
    role: str,
) -> dict[str, object]:
    identity_keys = (
        "evaluation_id",
        "task_id",
        "split",
        "source_episode_id",
        "reset_identity",
        "checkpoint_identity",
        "execution_horizon",
    )
    if any(correct.get(key) != intervention.get(key) for key in identity_keys):
        raise RuntimeError("task-intervention pair is not identity matched")
    if correct.get("task_condition_id") != correct.get("task_id"):
        raise RuntimeError("correct-condition baseline is mislabeled")
    if role == "wrong" and intervention.get("task_condition_id") == intervention.get("task_id"):
        raise RuntimeError("wrong-condition intervention did not change the Task ID")
    correct_actions = _actions(correct)
    intervention_actions = _actions(intervention)
    common = min(len(correct_actions), len(intervention_actions))
    if common < 1:
        raise RuntimeError("task-intervention pair submitted no comparable action")
    difference = intervention_actions[:common] - correct_actions[:common]
    first_l2 = float(np.linalg.norm(difference[0]))
    common_mean_l2 = float(np.linalg.norm(difference, axis=1).mean())
    behavior_difference = (
        correct.get("outcome") != intervention.get("outcome")
        or correct.get("final_evaluation") != intervention.get("final_evaluation")
        or not math.isclose(
            _optional_float(correct.get("maximum_object_motion_m")),
            _optional_float(intervention.get("maximum_object_motion_m")),
            abs_tol=1e-6,
        )
    )
    return {
        "evaluation_id": correct["evaluation_id"],
        "task_id": correct["task_id"],
        "role": role,
        "correct_task_condition_id": correct["task_condition_id"],
        "intervention_task_condition_id": intervention["task_condition_id"],
        "first_action_l2": first_l2,
        "common_prefix_action_count": common,
        "common_prefix_mean_action_l2": common_mean_l2,
        "action_changed": first_l2 > 1e-8 or common_mean_l2 > 1e-8,
        "behavior_changed": behavior_difference,
        "correct_outcome": correct["outcome"],
        "intervention_outcome": intervention["outcome"],
        "correct_success": correct["success"],
        "intervention_success": intervention["success"],
        "correct_wrong_task_behavior": correct.get("wrong_task_behavior"),
        "intervention_wrong_task_behavior": intervention.get("wrong_task_behavior"),
        "correct_maximum_object_motion_m": correct.get("maximum_object_motion_m"),
        "intervention_maximum_object_motion_m": intervention.get("maximum_object_motion_m"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.evaluation_root.resolve()
    records: dict[str, dict[str, list[dict[str, object]]]] = {}
    for role in ("correct", "wrong", "shuffled"):
        records[role] = {}
        for task_id in TASK_IDS:
            records[role][task_id] = _read_jsonl(root / role / task_id / "episodes.jsonl")
    comparisons: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        correct_rows = records["correct"][task_id]
        correct_by_id = {str(row["evaluation_id"]): row for row in correct_rows}
        for role in ("wrong", "shuffled"):
            rows = records[role][task_id]
            if [row["evaluation_id"] for row in rows] != [
                row["evaluation_id"] for row in correct_rows
            ]:
                raise RuntimeError("task-intervention roles use different schedule prefixes")
            comparisons.extend(
                _comparison(correct_by_id[str(row["evaluation_id"])], row, role=role)
                for row in rows
            )
    aggregate: dict[str, dict[str, object]] = {}
    for role in ("wrong", "shuffled"):
        selected = [row for row in comparisons if row["role"] == role]
        aggregate[role] = {
            "pair_count": len(selected),
            "action_changed_count": sum(bool(row["action_changed"]) for row in selected),
            "behavior_changed_count": sum(bool(row["behavior_changed"]) for row in selected),
            "mean_first_action_l2": statistics.fmean(
                _optional_float(row["first_action_l2"]) for row in selected
            ),
            "mean_common_prefix_action_l2": statistics.fmean(
                _optional_float(row["common_prefix_mean_action_l2"]) for row in selected
            ),
            "correct_success_count": sum(bool(row["correct_success"]) for row in selected),
            "intervention_success_count": sum(
                bool(row["intervention_success"]) for row in selected
            ),
            "intervention_wrong_task_behavior_count": sum(
                row["intervention_wrong_task_behavior"] is True for row in selected
            ),
        }
    semantic = {
        "schema_version": "langmani-v2-phase2c-a1-task-intervention-analysis-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "split": "validation",
        "roles": ["correct", "wrong", "shuffled"],
        "comparison_contract": (
            "same checkpoint, reset identity, source episode, execution horizon, "
            "and evaluator; Task ID alone changes"
        ),
        "comparisons": comparisons,
        "aggregate": aggregate,
        "task_id_action_effect_observed": any(
            _integer(value["action_changed_count"]) > 0 for value in aggregate.values()
        ),
        "task_id_behavior_effect_observed": any(
            _integer(value["behavior_changed_count"]) > 0 for value in aggregate.values()
        ),
        "language_grounding_claimed": False,
        "diagnostic_completion_requires_positive_effect": False,
        "passed": all(
            _integer(value["pair_count"]) == len(TASK_IDS) * 5 for value in aggregate.values()
        ),
    }
    result = {**semantic, "fingerprint": f"sha256:{sha256_hex(semantic)}"}
    destination = args.report.resolve()
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
