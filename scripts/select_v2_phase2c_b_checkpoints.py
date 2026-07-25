"""Select at most two SmolVLA checkpoints for bounded validation rollouts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import Phase2CBContractError, canonical_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2CBContractError(f"{path} must contain one object")
    semantic = dict(value)
    fingerprint = semantic.pop("fingerprint", None)
    if fingerprint != canonical_fingerprint(semantic):
        raise Phase2CBContractError(f"diagnostic fingerprint changed: {path}")
    return value


def _finite_metric(document: Mapping[str, object], key: str) -> float:
    value = document.get(key)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise Phase2CBContractError(f"diagnostic lacks finite {key}")
    return float(value)


def _optimizer_step(document: Mapping[str, object]) -> int:
    value = document.get("optimizer_step")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise Phase2CBContractError("diagnostic lacks a valid optimizer step")
    return value


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CBContractError(f"existing immutable selection differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    paths = [path.resolve() for path in args.diagnostic]
    if not 1 <= len(paths) <= 3 or len(paths) != len(set(paths)):
        raise Phase2CBContractError("checkpoint selection requires one to three unique diagnostics")
    documents = [_read_object(path) for path in paths]
    model_kinds = {str(document.get("model_kind")) for document in documents}
    checkpoint_hashes = {str(document.get("checkpoint_sha256")) for document in documents}
    optimizer_steps = {document.get("optimizer_step") for document in documents}
    if (
        len(model_kinds) != 1
        or len(checkpoint_hashes) != len(documents)
        or len(optimizer_steps) != len(documents)
        or any(
            document.get("schema_version")
            != "langmani-v2-phase2c-b-offline-checkpoint-diagnostic-v0"
            or document.get("package_fingerprint") != PACKAGE_FINGERPRINT
            or document.get("split") != "validation"
            or document.get("passed") is not True
            or document.get("invalid_action_count") != 0
            for document in documents
        )
    ):
        raise Phase2CBContractError("offline checkpoint diagnostics are incompatible")

    ranked = sorted(
        zip(paths, documents, strict=True),
        key=lambda item: (
            _finite_metric(item[1], "physical_first_action_mae"),
            _finite_metric(item[1], "arm_action_mae"),
            _finite_metric(item[1], "gripper_action_mae"),
            _finite_metric(item[1], "validation_flow_matching_loss"),
            _optimizer_step(item[1]),
        ),
    )
    selected = ranked[:2]
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-checkpoint-rollout-selection-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": next(iter(model_kinds)),
        "selection_split": "validation",
        "selection_count": len(selected),
        "maximum_selected_checkpoints": 2,
        "ranking_rule": [
            "minimize_physical_first_action_mae",
            "minimize_arm_action_mae",
            "minimize_gripper_action_mae",
            "minimize_validation_flow_matching_loss",
            "prefer_earlier_optimizer_step",
        ],
        "selected": [
            {
                "rank": rank,
                "checkpoint": document["checkpoint"],
                "checkpoint_sha256": document["checkpoint_sha256"],
                "optimizer_step": document["optimizer_step"],
                "diagnostic_path": path.as_posix(),
                "diagnostic_fingerprint": document["fingerprint"],
            }
            for rank, (path, document) in enumerate(selected, start=1)
        ],
        "all_candidates": [
            {
                "checkpoint_sha256": document["checkpoint_sha256"],
                "optimizer_step": document["optimizer_step"],
                "physical_first_action_mae": document["physical_first_action_mae"],
                "arm_action_mae": document["arm_action_mae"],
                "gripper_action_mae": document["gripper_action_mae"],
                "validation_flow_matching_loss": document["validation_flow_matching_loss"],
            }
            for _, document in ranked
        ],
        "selected_on_total_loss_only": False,
        "final_test_opened": False,
        "passed": True,
    }
    result = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
    _write_new_or_equal(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
