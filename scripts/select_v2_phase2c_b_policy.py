"""Select one SmolVLA checkpoint and execution horizon on a frozen validation subset."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import EXECUTION_HORIZONS, PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import Phase2CBContractError, canonical_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_fingerprinted(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2CBContractError(f"{path} must contain one object")
    semantic = dict(value)
    fingerprint = semantic.pop("fingerprint", None)
    if fingerprint not in {
        canonical_fingerprint(semantic),
        f"sha256:{sha256_hex(semantic)}",
    }:
        raise Phase2CBContractError(f"artifact fingerprint changed: {path}")
    return value


def _metric(document: Mapping[str, object], key: str) -> float:
    value = document.get(key)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise Phase2CBContractError(f"validation summary lacks finite {key}")
    return float(value)


def _integer(document: Mapping[str, object], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase2CBContractError(f"validation summary lacks valid {key}")
    return value


def _outcome(document: Mapping[str, object], key: str) -> int:
    outcomes = document.get("outcomes")
    value = outcomes.get(key, 0) if isinstance(outcomes, Mapping) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase2CBContractError(f"validation summary has invalid outcome {key}")
    return value


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CBContractError(f"existing immutable policy selection differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    checkpoint_selection = _read_fingerprinted(args.checkpoint_selection.resolve())
    selected_checkpoints = checkpoint_selection.get("selected")
    if (
        checkpoint_selection.get("schema_version")
        != "langmani-v2-phase2c-b-checkpoint-rollout-selection-v0"
        or checkpoint_selection.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or checkpoint_selection.get("passed") is not True
        or not isinstance(selected_checkpoints, list)
        or not 1 <= len(selected_checkpoints) <= 2
    ):
        raise Phase2CBContractError("checkpoint rollout selection is invalid")
    checkpoint_ranks = {
        str(item["checkpoint_sha256"]): int(item["rank"])
        for item in selected_checkpoints
        if isinstance(item, Mapping)
    }
    if len(checkpoint_ranks) != len(selected_checkpoints):
        raise Phase2CBContractError("selected checkpoint registry is malformed")

    paths = [path.resolve() for path in args.summary]
    summaries = [_read_fingerprinted(path) for path in paths]
    expected_count = len(checkpoint_ranks) * len(EXECUTION_HORIZONS)
    if len(summaries) != expected_count:
        raise Phase2CBContractError("policy selection lacks one checkpoint/horizon pair")
    candidates: list[dict[str, object]] = []
    observed_pairs: set[tuple[str, int]] = set()
    common_episode_count: int | None = None
    model_kind = str(checkpoint_selection.get("model_kind"))
    task_id: str | None = None
    for path, summary in zip(paths, summaries, strict=True):
        identities = summary.get("checkpoint_identities")
        checkpoint_sha256 = (
            str(identities[0]) if isinstance(identities, list) and len(identities) == 1 else ""
        )
        horizon = summary.get("execution_horizon")
        episode_count = summary.get("episode_count")
        if (
            summary.get("schema_version") != "langmani-v2-phase2c-a-evaluation-summary-v0"
            or summary.get("package_fingerprint") != PACKAGE_FINGERPRINT
            or summary.get("model_kind") != model_kind
            or summary.get("split") != "validation"
            or summary.get("instruction_condition") != "correct"
            or checkpoint_sha256 not in checkpoint_ranks
            or not isinstance(horizon, int)
            or isinstance(horizon, bool)
            or horizon not in EXECUTION_HORIZONS
            or not isinstance(episode_count, int)
            or episode_count < 1
        ):
            raise Phase2CBContractError("validation summary is outside policy selection scope")
        pair = (checkpoint_sha256, horizon)
        if pair in observed_pairs:
            raise Phase2CBContractError("duplicate checkpoint/horizon validation summary")
        observed_pairs.add(pair)
        common_episode_count = (
            episode_count if common_episode_count is None else common_episode_count
        )
        if episode_count != common_episode_count:
            raise Phase2CBContractError("horizon candidates use different episode counts")
        current_task = str(summary.get("task_id"))
        task_id = current_task if task_id is None else task_id
        if current_task != task_id:
            raise Phase2CBContractError("policy selection summaries mix tasks")
        candidates.append(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_offline_rank": checkpoint_ranks[checkpoint_sha256],
                "execution_horizon": horizon,
                "episode_count": episode_count,
                "success_count": _integer(summary, "success_count"),
                "success_rate": _metric(summary, "success_rate"),
                "invalid_action_count": _integer(summary, "invalid_action_count"),
                "simulator_error_count": _integer(summary, "simulator_error_count"),
                "timeout_count": _outcome(summary, "timeout"),
                "mean_action_smoothness_l2": summary.get("mean_action_smoothness_l2"),
                "mean_policy_query_count": _metric(summary, "mean_policy_query_count"),
                "inference_latency_p95_ms": summary.get("inference_latency_p95_ms"),
                "summary_path": path.as_posix(),
                "summary_fingerprint": summary["fingerprint"],
            }
        )
    expected_pairs = {
        (checkpoint, horizon) for checkpoint in checkpoint_ranks for horizon in EXECUTION_HORIZONS
    }
    if observed_pairs != expected_pairs:
        raise Phase2CBContractError("checkpoint/horizon validation grid is incomplete")
    eligible = [
        candidate
        for candidate in candidates
        if candidate["invalid_action_count"] == 0 and candidate["simulator_error_count"] == 0
    ]
    if not eligible:
        raise Phase2CBContractError("all validation candidates are invalid")

    def optional_metric(candidate: Mapping[str, object], key: str) -> float:
        value = candidate.get(key)
        return float(value) if isinstance(value, int | float) else math.inf

    def candidate_integer(candidate: Mapping[str, object], key: str) -> int:
        value = candidate.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise Phase2CBContractError(f"policy candidate lacks integer {key}")
        return value

    def candidate_metric(candidate: Mapping[str, object], key: str) -> float:
        value = candidate.get(key)
        if not isinstance(value, int | float):
            raise Phase2CBContractError(f"policy candidate lacks numeric {key}")
        return float(value)

    eligible.sort(
        key=lambda candidate: (
            -candidate_metric(candidate, "success_rate"),
            candidate_integer(candidate, "timeout_count"),
            optional_metric(candidate, "mean_action_smoothness_l2"),
            candidate_metric(candidate, "mean_policy_query_count"),
            optional_metric(candidate, "inference_latency_p95_ms"),
            candidate_integer(candidate, "checkpoint_offline_rank"),
            candidate_integer(candidate, "execution_horizon"),
        )
    )
    selected = eligible[0]
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-selected-policy-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": model_kind,
        "task_id": task_id,
        "selection_split": "validation",
        "selection_subset_episode_count": common_episode_count,
        "checkpoint_selection_fingerprint": checkpoint_selection["fingerprint"],
        "selection_rule": (
            "maximize success; then minimize timeout, action smoothness, policy queries, "
            "p95 latency, offline checkpoint rank, and execution horizon; invalid actions "
            "or simulator errors are ineligible"
        ),
        "candidates": candidates,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_execution_horizon": selected["execution_horizon"],
        "selected_summary_fingerprint": selected["summary_fingerprint"],
        "final_test_opened": False,
        "passed": True,
    }
    result = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
    _write_new_or_equal(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
