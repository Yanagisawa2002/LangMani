"""Independently verify the compact Phase 2C-C decision package."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

PACKAGE_FINGERPRINT = "sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04"


class Phase2CCContractError(RuntimeError):
    """Raised when compact evidence violates the independent contract."""


def canonical_fingerprint(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


BASE_FILES = {
    "input_verification.json",
    "state_action_audit.json",
    "residual_transform_manifest.json",
    "action_100k_audit.json",
    "reconstruction_audit.json",
    "baseline_phase_diagnostics.json",
    "absolute_training_reset_evaluation.json",
    "relative_training_config.json",
    "smoke_micro_result.json",
    "training_metrics.json",
    "checkpoint_registry.jsonl",
    "validation_result.json",
    "relative_training_reset_result.json",
    "result_classification.json",
    "next_route_decision.json",
    "authorization_state.json",
    "artifact_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CCContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CCContractError(f"{path} must contain one object")
    semantic = dict(value)
    fingerprint = semantic.pop("fingerprint", None)
    if fingerprint != canonical_fingerprint(semantic):
        raise Phase2CCContractError(f"fingerprint changed: {path}")
    return value


def _write(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise Phase2CCContractError(f"existing verification differs: {path}")
    if not path.exists():
        path.write_text(encoded, encoding="utf-8", newline="\n")


def _integer(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        raise Phase2CCContractError(f"required integer is missing: {key}")
    return result


def _expected_case(validation_success: int, training_success: int) -> str:
    if validation_success >= 3:
        return "CASE_A"
    if training_success >= 1:
        return "CASE_B"
    if validation_success == 0:
        return "CASE_C"
    raise Phase2CCContractError("compact evidence falls outside the frozen decision matrix")


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    manifest = _read(root / "artifact_manifest.json")
    case = str(manifest.get("case"))
    expected_files = set(BASE_FILES)
    if case == "CASE_A":
        expected_files.add("unseen_reset_result.json")
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise Phase2CCContractError(
            f"compact artifact file allowlist changed: {sorted(actual_files ^ expected_files)}"
        )
    records = manifest.get("records")
    if (
        manifest.get("schema_version") != "langmani-v2-phase2c-c-artifact-manifest-v0"
        or manifest.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or manifest.get("passed") is not True
        or not isinstance(records, list)
        or manifest.get("file_count") != len(expected_files) - 1
        or len(records) != len(expected_files) - 1
    ):
        raise Phase2CCContractError("artifact manifest contract changed")
    total_bytes = 0
    observed_paths: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise Phase2CCContractError("artifact manifest contains a non-object record")
        relative = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or relative in observed_paths
            or "/" in relative
            or "\\" in relative
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not isinstance(digest, str)
        ):
            raise Phase2CCContractError("artifact manifest record is malformed")
        path = root / relative
        if not path.is_file() or path.stat().st_size != size or _sha256(path) != digest:
            raise Phase2CCContractError(f"compact artifact bytes changed: {relative}")
        observed_paths.add(relative)
        total_bytes += size
    if (
        observed_paths != expected_files - {"artifact_manifest.json"}
        or manifest.get("total_bytes") != total_bytes
    ):
        raise Phase2CCContractError("artifact manifest coverage changed")

    documents = {
        name: _read(root / name)
        for name in expected_files
        if name not in {"artifact_manifest.json", "checkpoint_registry.jsonl"}
    }
    if any(
        value.get("package_fingerprint") != PACKAGE_FINGERPRINT
        for name, value in documents.items()
        if name != "action_100k_audit.json"
    ):
        raise Phase2CCContractError("compact document package identity changed")
    transform = documents["residual_transform_manifest.json"]
    static_audit = documents["action_100k_audit.json"]
    reconstruction = documents["reconstruction_audit.json"]
    smoke_micro = documents["smoke_micro_result.json"]
    training = documents["training_metrics.json"]
    absolute_train = documents["absolute_training_reset_evaluation.json"]
    validation_result = documents["validation_result.json"]
    relative_train = documents["relative_training_reset_result.json"]
    classification = documents["result_classification.json"]
    decision = documents["next_route_decision.json"]
    authorization = documents["authorization_state.json"]
    if (
        transform.get("schema_version")
        != "langmani-v2-phase2c-c-pick-state-relative-bounded-residual-v1"
        or transform.get("post_hoc_clipping") is not False
        or transform.get("projection") is not False
        or transform.get("replacement_action") is not False
        or static_audit.get("passed") is not True
        or static_audit.get("chunk_count") != 100_000
        or static_audit.get("nonfinite_component_count") != 0
        or static_audit.get("lower_bound_violation_count") != 0
        or static_audit.get("upper_bound_violation_count") != 0
        or static_audit.get("transform_fingerprint") != transform.get("fingerprint")
        or reconstruction.get("passed") is not True
        or float(reconstruction.get("maximum_absolute_error", 1.0)) > 1e-6
        or reconstruction.get("over_tolerance_count") != 0
        or smoke_micro.get("passed") is not True
        or smoke_micro.get("real_environment_steps_completed") != 2
        or training.get("passed") is not True
    ):
        raise Phase2CCContractError("transform or training gate evidence changed")
    full_training = training.get("full_training")
    diagnostics = training.get("offline_diagnostics")
    selection = training.get("checkpoint_selection")
    if (
        not isinstance(full_training, Mapping)
        or full_training.get("optimizer_steps") != 20_000
        or not isinstance(diagnostics, list)
        or {row.get("optimizer_step") for row in diagnostics if isinstance(row, Mapping)}
        != {5_000, 10_000, 20_000}
        or not isinstance(selection, Mapping)
        or selection.get("selected_optimizer_step") not in {5_000, 10_000, 20_000}
    ):
        raise Phase2CCContractError("full-training checkpoint evidence changed")
    registry = [
        json.loads(line)
        for line in (root / "checkpoint_registry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [row.get("optimizer_step") for row in registry] != [5_000, 10_000, 20_000]:
        raise Phase2CCContractError("compact checkpoint registry changed")
    validation = validation_result.get("validation")
    if not isinstance(validation, Mapping):
        raise Phase2CCContractError("validation result lacks its summary")
    validation_success = _integer(validation, "success_count")
    training_success = _integer(relative_train, "success_count")
    if (
        _integer(validation, "episode_count") != 30
        or _integer(relative_train, "episode_count") != 30
        or _integer(absolute_train, "episode_count") != 30
        or classification.get("case") != _expected_case(validation_success, training_success)
        or classification.get("case") != case
        or decision.get("case") != case
        or authorization.get("case") != case
    ):
        raise Phase2CCContractError("A/B/C result classification changed")
    if (
        authorization.get("vla_jepa_authorized") is not False
        or authorization.get("vla_jepa_training_started") is not False
        or authorization.get("shared_smolvla_training_started") is not False
        or authorization.get("stack_smolvla_training_started") is not False
        or authorization.get("push_smolvla_training_started") is not False
        or authorization.get("phase2c_c_1_authorized") is not False
    ):
        raise Phase2CCContractError("post-result authorization boundary changed")
    unseen_verified = False
    if case == "CASE_A":
        unseen = documents["unseen_reset_result.json"]
        unseen_verified = (
            unseen.get("split_role") == "test_unseen_reset"
            and unseen.get("episode_count") == 50
            and validation_success >= 3
        )
        if not unseen_verified:
            raise Phase2CCContractError("Case A unseen-reset evidence changed")

    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-independent-verification-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "artifact_manifest_fingerprint": manifest["fingerprint"],
        "verified_file_count": len(records),
        "verified_total_bytes": total_bytes,
        "transform_exact_and_bounded": True,
        "full_training_verified": True,
        "validation_success_count": validation_success,
        "training_reset_success_count": training_success,
        "case": case,
        "unseen_reset_verified": unseen_verified,
        "vla_jepa_unauthorized": True,
        "passed": True,
    }
    result = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
    if args.output is not None:
        _write(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
