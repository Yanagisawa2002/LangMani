"""Freeze the validation-only Task-ID intervention assignments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT, TASK_IDS


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes_per_task < 1 or args.seed < 0:
        raise ValueError("intervention size and seed must be positive")
    schedule = _read_object(args.schedule.resolve())
    if schedule.get("package_fingerprint") != PACKAGE_FINGERPRINT:
        raise RuntimeError("evaluation schedule uses a noncanonical package")
    schedules = schedule.get("schedules")
    validation = schedules.get("validation") if isinstance(schedules, dict) else None
    if not isinstance(validation, dict):
        raise RuntimeError("evaluation schedule lacks validation identities")
    rows_by_task: dict[str, list[dict[str, object]]] = {}
    interleaved: list[tuple[str, dict[str, object]]] = []
    for index in range(args.episodes_per_task):
        for task_id in TASK_IDS:
            task_rows = validation.get(task_id)
            if not isinstance(task_rows, list) or len(task_rows) < args.episodes_per_task:
                raise RuntimeError("validation schedule is too small for the intervention")
            row = task_rows[index]
            if not isinstance(row, dict) or row.get("task_id") != task_id:
                raise RuntimeError("validation schedule row is malformed")
            rows_by_task.setdefault(task_id, []).append(row)
            interleaved.append((task_id, row))
    correct_labels = [task_id for task_id, _ in interleaved]
    shuffled_labels = list(correct_labels)
    attempt = 0
    while True:
        random.Random(args.seed + attempt).shuffle(shuffled_labels)
        if shuffled_labels != correct_labels and all(
            len(
                {
                    shuffled_labels[index]
                    for index, (row_task, _) in enumerate(interleaved)
                    if row_task == task_id
                }
            )
            >= 2
            for task_id in TASK_IDS
        ):
            break
        shuffled_labels = list(correct_labels)
        attempt += 1
        if attempt > 10_000:
            raise RuntimeError("could not construct a nondegenerate shuffled intervention")
    shuffled_by_id = {
        str(row["evaluation_id"]): shuffled_labels[index]
        for index, (_, row) in enumerate(interleaved)
    }
    assignments_by_task = {
        task_id: [
            {
                "evaluation_id": str(row["evaluation_id"]),
                "correct": task_id,
                "wrong": TASK_IDS[(TASK_IDS.index(task_id) + 1) % len(TASK_IDS)],
                "shuffled": shuffled_by_id[str(row["evaluation_id"])],
                "reset_identity": row["reset_identity"],
            }
            for row in rows_by_task[task_id]
        ]
        for task_id in TASK_IDS
    }
    semantic = {
        "schema_version": "langmani-v2-phase2c-a1-task-intervention-lock-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "schedule_fingerprint": schedule.get("fingerprint"),
        "split": "validation",
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "shuffle_attempt": attempt,
        "wrong_rule": "next canonical task ID cyclically",
        "shuffled_rule": (
            "seeded permutation of the balanced correct-label vector; "
            "at least two assigned labels per true task"
        ),
        "assignments_by_task": assignments_by_task,
        "final_test_identities_accessed": False,
    }
    result = {**semantic, "fingerprint": f"sha256:{sha256_hex(semantic)}"}
    destination = args.report.resolve()
    if destination.exists():
        if _read_object(destination) != result:
            raise RuntimeError(f"existing immutable artifact differs: {destination}")
    else:
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
