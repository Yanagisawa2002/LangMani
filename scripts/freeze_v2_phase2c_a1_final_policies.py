"""Freeze ACT checkpoints, processors, horizon, evaluator, and final identities."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import (
    FINAL_EPISODES_PER_TASK,
    PACKAGE_FINGERPRINT,
    TASK_IDS,
    ModelKind,
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        action="append",
        metavar="MODEL_KIND=PATH",
        required=True,
    )
    parser.add_argument("--horizon-selection", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--evaluation-git-commit", required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _selections(values: list[str]) -> dict[str, dict[str, object]]:
    expected = {kind.value for kind in ModelKind}
    result = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or name in result:
            raise RuntimeError("--selection must uniquely use MODEL_KIND=PATH")
        document = _read_object(Path(raw_path).resolve())
        selected = document.get("selected")
        if (
            document.get("model_kind") != name
            or document.get("selection_split") != "validation"
            or document.get("test_outcomes_available") is not False
            or document.get("passed") is not True
            or not isinstance(selected, list)
            or len(selected) != 1
            or not isinstance(selected[0], dict)
        ):
            raise RuntimeError(f"{name} selection is not one validation-only checkpoint")
        result[name] = {
            "selection_fingerprint": document.get("fingerprint"),
            **selected[0],
        }
    if set(result) != expected:
        raise RuntimeError("final lock requires Pick, Stack, Push, and shared selections")
    return result


def main() -> int:
    args = parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.evaluation_git_commit) is None:
        raise RuntimeError("evaluation Git commit must be one full SHA")
    policies = _selections(args.selection)
    horizon = _read_object(args.horizon_selection.resolve())
    selected_horizon = horizon.get("selected_execution_horizon")
    if (
        horizon.get("selection_split") != "validation"
        or horizon.get("final_test_outcomes_available") is not False
        or selected_horizon not in {1, 4, 8}
        or horizon.get("passed") is not True
    ):
        raise RuntimeError("execution-horizon selection is not final eligible")
    schedule = _read_object(args.schedule.resolve())
    schedules = schedule.get("schedules")
    if schedule.get("package_fingerprint") != PACKAGE_FINGERPRINT or not isinstance(
        schedules, dict
    ):
        raise RuntimeError("final schedule identity is invalid")
    final_identity_counts: dict[str, dict[str, int]] = {}
    for split in ("test_unseen_reset", "test_visual_shift"):
        split_value = schedules.get(split)
        if not isinstance(split_value, dict):
            raise RuntimeError(f"final schedule lacks {split}")
        final_identity_counts[split] = {}
        for task_id in TASK_IDS:
            rows = split_value.get(task_id)
            if not isinstance(rows, list) or len(rows) != FINAL_EPISODES_PER_TASK:
                raise RuntimeError(f"final schedule count changed for {split}/{task_id}")
            final_identity_counts[split][task_id] = len(rows)
    semantic = {
        "schema_version": "langmani-v2-phase2c-a1-final-policy-lock-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "schedule_fingerprint": schedule.get("fingerprint"),
        "evaluation_git_commit": args.evaluation_git_commit,
        "policies": policies,
        "execution_horizon": selected_horizon,
        "execution_horizon_selection_fingerprint": horizon.get("fingerprint"),
        "maximum_steps": 200,
        "action_handling": "hard_reject_nonfinite_malformed_or_out_of_bounds",
        "action_clipping": False,
        "action_projection": False,
        "task_conditioning": {
            "per_task": "none",
            "shared_act": "three_way_onehot_public_env_token",
        },
        "final_identity_counts": final_identity_counts,
        "settings_mutable_after_final_results": False,
        "final_results_available": False,
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
