"""Integrity-check and summarize one local M4 ACT checkpoint."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from langmani.policies.act_checkpoint import CHECKPOINT_MANIFEST, load_act_checkpoint
from langmani.policies.act_runtime import atomic_write_json
from langmani.policies.act_types import ActRunIdentity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m4" / "checkpoint_inspection.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def inspect_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = path.resolve()
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("checkpoint must be one direct child of a run's checkpoints directory")
    run_root = checkpoint.parent.parent
    relative = checkpoint.relative_to(run_root).as_posix()
    manifest = _read_object(checkpoint / CHECKPOINT_MANIFEST)
    identity_value = manifest.get("identity")
    if not isinstance(identity_value, dict):
        raise RuntimeError("checkpoint manifest is missing its run identity")
    identity = ActRunIdentity.from_dict(identity_value)
    loaded = load_act_checkpoint(
        run_root=run_root,
        checkpoint_relative_path=relative,
        expected_identity=identity,
    )
    policy = loaded.policy
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    return {
        "schema_version": "langmani-m4-checkpoint-inspection-v1",
        "passed": True,
        "run_fingerprint": identity.run_fingerprint,
        "checkpoint": loaded.record.to_dict(),
        "training_state": loaded.training_state.to_dict(),
        "variant": identity.variant.value,
        "task_id": identity.task_id,
        "parameter_count": parameter_count,
        "local_strict_reload": True,
        "processor_reload": True,
    }


def main() -> int:
    args = parse_args()
    try:
        report = inspect_checkpoint(args.checkpoint)
    except Exception as error:  # noqa: BLE001 - CLI boundary preserves diagnostics
        traceback.print_exc()
        report = {
            "schema_version": "langmani-m4-checkpoint-inspection-v1",
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
        }
    atomic_write_json(args.report, report)
    print(json.dumps({**report, "report": str(args.report.resolve())}, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
