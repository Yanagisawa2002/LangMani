"""Fresh-process independent verifier for the M5A sealed final benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from langmani.datasets.identity import canonical_json, sha256_hex
from langmani.language.artifact_validation import validate_development_control_evidence
from langmani.language.controller_registry import ControllerRegistry
from langmani.language.corpus import (
    build_language_corpus,
    materialize_final_language_examples,
    materialize_final_language_slot_map,
)
from langmani.language.neuro_symbolic_dispatch import (
    validate_selected_neuro_symbolic_dispatch_source,
)
from langmani.language.neuro_symbolic_experiment import compute_neuro_symbolic_safety_metrics
from langmani.language.neuro_symbolic_safety import (
    evaluate_router_contracts,
    special_safety_route_counts,
)
from langmani.language.offline_router_metrics import (
    recompute_router_metrics,
    router_record_from_dict,
)
from langmani.language.router_types import LanguageExample, LanguageSplit
from langmani.language.schedules import build_language_schedule_locks, build_staged_control_schedule
from langmani.language.sealed_final import (
    SEALED_FINAL_VERIFICATION_SCHEMA,
    M5AFinalRunIdentityV0,
    analyze_final_control_records,
    build_final_result,
    evaluate_final_language_quality,
    final_metric_contract,
    validate_final_authorization,
)
from langmani.language.sealed_final_evidence import (
    final_access_count,
    read_object,
    real_unlinked,
    validate_sealed_final_evidence,
)
from langmani.language.stage_protocol import M5AStage
from langmani.policies.act_runtime import atomic_write_json
from scripts.evaluate_neuro_symbolic_router import (
    DEFAULT_CLASSIFIER_CHECKPOINT,
    DEFAULT_CLASSIFIER_REJECTION_EVIDENCE,
    DEFAULT_QWEN4B_EVIDENCE,
    DEFAULT_QWEN17B_EVIDENCE,
    _archive_fingerprint_prelock,
    _frozen_baseline_identity,
    _load_classifier_negative,
    _semantic_field_metrics,
)
from scripts.run_language_control import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUNTIME_SELECTION,
    DevelopmentControlInputs,
    _load_frozen_controller_registry,
    load_authoritative_final_schedule,
)
from scripts.run_m5a_sealed_final import (
    DEFAULT_CORPUS_ROOT,
    DEFAULT_NEURO_EVIDENCE,
    DEFAULT_SCHEDULE,
    EXPECTED_CONTROL_FINAL_SCHEDULE_FINGERPRINT,
    EXPECTED_CONTROLLER_REGISTRY_FINGERPRINT,
    EXPECTED_FINAL_AUTHORIZATION_FINGERPRINT,
    EXPECTED_LANGUAGE_FINAL_SCHEDULE_FINGERPRINT,
    EXPECTED_ROUTER_LOCK_FINGERPRINT,
    EXPECTED_RUNTIME_FINGERPRINT,
)

ROUTER_ARTIFACTS = {
    "rule_router": "language/rule_router.json",
    "classifier_negative_baseline": "language/classifier_negative_baseline.json",
    "qwen17b_negative_baseline": "language/qwen17b_negative_baseline.json",
    "qwen4b_negative_baseline": "language/qwen4b_negative_baseline.json",
    "neuro_symbolic_final": "language/neuro_symbolic_final.json",
}


class M5AFinalVerificationError(RuntimeError):
    """Raised when sealed-final evidence cannot be independently reproduced."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--sealed-output-root", type=Path)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument(
        "--classifier-rejection-evidence",
        type=Path,
        default=DEFAULT_CLASSIFIER_REJECTION_EVIDENCE,
    )
    parser.add_argument("--qwen17b-evidence-root", type=Path, default=DEFAULT_QWEN17B_EVIDENCE)
    parser.add_argument("--qwen4b-evidence-root", type=Path, default=DEFAULT_QWEN4B_EVIDENCE)
    parser.add_argument(
        "--neuro-symbolic-evidence-root",
        type=Path,
        default=DEFAULT_NEURO_EVIDENCE,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--runtime-selection", type=Path, default=DEFAULT_RUNTIME_SELECTION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/diagnostics/m5a/sealed-final-independent-verification.json"),
    )
    parser.add_argument("--structural", action="store_true")
    return parser


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise M5AFinalVerificationError(f"{label} must be one JSON object")
    return cast(dict[str, object], value)


def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise M5AFinalVerificationError(f"{label} must be one JSON array")
    return cast(list[object], value)


def _equal(observed: object, expected: object, *, label: str) -> None:
    if canonical_json(observed) != canonical_json(expected):
        raise M5AFinalVerificationError(f"{label} differs from independent recomputation")


def _records(
    payload: Mapping[str, object], *, examples: Sequence[LanguageExample]
) -> tuple[tuple[object, ...], Mapping[str, object]]:
    raw = _sequence(payload.get("records"), label="router records")
    if len(raw) != len(examples):
        raise M5AFinalVerificationError("router record count differs from final schedule")
    parsed = tuple(router_record_from_dict(_object(value, label="router record")) for value in raw)
    metrics = recompute_router_metrics(records=parsed, examples=examples)
    _equal(payload.get("metrics"), metrics, label="router metrics")
    return parsed, metrics


def _verify_language(
    *, root: Path, examples: tuple[LanguageExample, ...]
) -> tuple[dict[str, object], dict[str, object]]:
    payloads: dict[str, object] = {}
    parsed_by_name: dict[str, tuple[object, ...]] = {}
    for name, relative in ROUTER_ARTIFACTS.items():
        payload = read_object(root / relative, label=name)
        if payload.get("split") != LanguageSplit.FINAL.value:
            raise M5AFinalVerificationError(f"{name} does not identify the final split")
        parsed, _metrics = _records(payload, examples=examples)
        parsed_by_name[name] = parsed
        payloads[name] = payload

    candidate = parsed_by_name["neuro_symbolic_final"]
    safety, taxonomy = evaluate_router_contracts(records=candidate, examples=examples)
    special = special_safety_route_counts(records=candidate, examples=examples)
    candidate_payload = cast(Mapping[str, object], payloads["neuro_symbolic_final"])
    _equal(candidate_payload.get("safety_evaluation"), safety.to_dict(), label="safety metrics")
    _equal(
        candidate_payload.get("taxonomy_evaluation"),
        taxonomy.to_dict(),
        label="taxonomy metrics",
    )
    metrics = cast(Mapping[str, object], candidate_payload["metrics"])
    family_rates = cast(Mapping[str, float], metrics["rejection_family_false_route_rates"])
    gate = evaluate_final_language_quality(
        metrics=metrics,
        safety=safety,
        taxonomy=taxonomy,
        special_route_counts=special,
        rejection_family_unsafe_false_route_rates=family_rates,
        no_prohibited_source_access=True,
    )
    stored_gate = read_object(root / "language_quality_gate.json", label="language gate")
    _equal(stored_gate, gate, label="language gate")
    system_metrics = read_object(root / "language/system_metrics.json", label="system metrics")
    expected_system = {
        **compute_neuro_symbolic_safety_metrics(records=candidate, examples=examples),
        **special,
        "constrained_schema_compilation_seconds": system_metrics.get(
            "constrained_schema_compilation_seconds"
        ),
        "semantic_field_metrics": _semantic_field_metrics(records=candidate, examples=examples),
    }
    _equal(system_metrics, expected_system, label="language system metrics")
    return payloads, gate


def _verify_attempt(
    *, sealed_output_root: Path, evidence: Mapping[str, object]
) -> dict[str, object]:
    if final_access_count(sealed_output_root) != 1:
        raise M5AFinalVerificationError("sealed final must have exactly one access attempt")
    ledger = real_unlinked(sealed_output_root, label="sealed output") / "access-ledger"
    attempts = [value for value in ledger.iterdir() if value.is_dir()]
    if len(attempts) != 1:
        raise M5AFinalVerificationError("final access ledger differs")
    opened = read_object(attempts[0] / "opened.json", label="final attempt owner")
    completed = read_object(attempts[0] / "completed.json", label="final attempt completion")
    if (
        opened.get("attempt_index") != 1
        or completed.get("completed") is not True
        or completed.get("evidence_fingerprint") != evidence.get("artifact_fingerprint")
        or (attempts[0] / "invalid.json").exists()
    ):
        raise M5AFinalVerificationError("final access attempt is not one valid completed attempt")
    return {"opened": opened, "completed": completed}


def _verify_target(args: argparse.Namespace) -> dict[str, object]:
    if args.evidence_root is None or args.sealed_output_root is None:
        raise M5AFinalVerificationError("target verification requires evidence and output roots")
    root = real_unlinked(args.evidence_root, label="sealed-final evidence")
    evidence = validate_sealed_final_evidence(root)
    owner = cast(Mapping[str, object], evidence["owner"])
    attempt = _verify_attempt(sealed_output_root=args.sealed_output_root, evidence=evidence)

    corpus = build_language_corpus()
    final_examples = materialize_final_language_examples(corpus, authorize_final=True)
    final_slots = materialize_final_language_slot_map(corpus, authorize_final=True)
    _language_development, language_final = build_language_schedule_locks(corpus)
    control_lock = load_authoritative_final_schedule(args.schedule, corpus=corpus)
    staged = build_staged_control_schedule(
        control_lock, stage=M5AStage.SEALED_FINAL, authorize_final=True
    )
    if (
        len(final_examples) != 600
        or len(staged.episodes) != 72
        or language_final.schedule_fingerprint != EXPECTED_LANGUAGE_FINAL_SCHEDULE_FINGERPRINT
        or staged.schedule_fingerprint != EXPECTED_CONTROL_FINAL_SCHEDULE_FINGERPRINT
    ):
        raise M5AFinalVerificationError("materialized final schedules differ from their locks")

    language_schedule = read_object(root / "language_schedule.json", label="language schedule")
    control_schedule = read_object(root / "control_schedule.json", label="control schedule")
    _equal(
        language_schedule.get("examples"),
        [value.to_dict() for value in final_examples],
        label="final language examples",
    )
    _equal(
        {key: language_schedule.get(key) for key in language_final.to_dict()},
        language_final.to_dict(),
        label="final language lock",
    )
    _equal(
        {key: control_schedule.get(key) for key in staged.to_dict()},
        staged.to_dict(),
        label="final control schedule",
    )
    if language_schedule.get("corpus_archive_fingerprint") != _archive_fingerprint_prelock(
        args.corpus_root, corpus
    ):
        raise M5AFinalVerificationError("final corpus archive identity differs")

    selected_source = validate_selected_neuro_symbolic_dispatch_source(
        args.neuro_symbolic_evidence_root, corpus=corpus
    )
    router_identity = read_object(root / "router_identity.json", label="router identity")
    _equal(router_identity, selected_source.to_dict(), label="selected router identity")
    if selected_source.router_lock_fingerprint != EXPECTED_ROUTER_LOCK_FINGERPRINT:
        raise M5AFinalVerificationError("selected router lock differs")

    registry_payload = read_object(root / "controller_registry.json", label="controller registry")
    registry = ControllerRegistry.from_dict(registry_payload)
    if registry.registry_fingerprint != EXPECTED_CONTROLLER_REGISTRY_FINGERPRINT:
        raise M5AFinalVerificationError("controller registry identity differs")
    live_registry, _locators = _load_frozen_controller_registry(
        checkpoint_root=args.checkpoint_root,
        dataset_root=args.dataset_root,
        runtime_selection_path=args.runtime_selection,
    )
    _equal(live_registry.to_dict(), registry.to_dict(), label="live controller registry")

    authorization = read_object(root / "final_authorization.json", label="final authorization")
    validate_final_authorization(
        authorization,
        expected_fingerprint=EXPECTED_FINAL_AUTHORIZATION_FINGERPRINT,
        router_fingerprint=EXPECTED_ROUTER_LOCK_FINGERPRINT,
        controller_registry_fingerprint=EXPECTED_CONTROLLER_REGISTRY_FINGERPRINT,
        runtime_fingerprint=EXPECTED_RUNTIME_FINGERPRINT,
        language_schedule_fingerprint=EXPECTED_LANGUAGE_FINAL_SCHEDULE_FINGERPRINT,
        control_schedule_fingerprint=EXPECTED_CONTROL_FINAL_SCHEDULE_FINGERPRINT,
    )
    run_identity = read_object(root / "final_run_identity.json", label="final run identity")
    metric_contract = final_metric_contract()
    reconstructed = M5AFinalRunIdentityV0(
        **{
            key: value
            for key, value in run_identity.items()
            if key not in {"schema_version", "run_fingerprint"}
        }
    )
    _equal(run_identity, reconstructed.to_dict(), label="final run identity")
    if (
        reconstructed.metric_contract_fingerprint != metric_contract["contract_fingerprint"]
        or owner.get("run_fingerprint") != reconstructed.run_fingerprint
        or cast(Mapping[str, object], attempt["opened"]).get("run_identity", {}) != run_identity
    ):
        raise M5AFinalVerificationError("run, metric, or access-ledger identity differs")

    _payloads, language_gate = _verify_language(root=root, examples=final_examples)
    model_identities = read_object(
        root / "language/model_identities.json", label="model identities"
    )
    classifier, classifier_identity = _load_classifier_negative(
        checkpoint=args.classifier_checkpoint,
        rejection_root=args.classifier_rejection_evidence,
        local_files_only=True,
    )
    if not classifier.verify_unchanged():
        raise M5AFinalVerificationError("frozen classifier weights changed during verification")
    expected_model_identities = {
        "classifier_negative_baseline": classifier_identity,
        "qwen17b_negative_baseline": _frozen_baseline_identity(
            args.qwen17b_evidence_root, label="Qwen3-1.7B"
        ),
        "qwen4b_negative_baseline": _frozen_baseline_identity(
            args.qwen4b_evidence_root, label="direct Qwen3-4B"
        ),
    }
    for key, value in expected_model_identities.items():
        _equal(model_identities.get(key), value, label=key)
    _equal(
        model_identities.get("model_file_identities"),
        dict(selected_source.model_file_identities),
        label="selected model files",
    )
    if model_identities.get("snapshot_size_bytes") != selected_source.snapshot_size_bytes:
        raise M5AFinalVerificationError("selected model snapshot size differs")
    runtime = read_object(root / "runtime_identity.json", label="runtime identity")
    raw_root = runtime.get("control_atom_evidence_root")
    if not isinstance(raw_root, str):
        raise M5AFinalVerificationError("runtime identity lacks raw control evidence")
    inputs = DevelopmentControlInputs(
        schedule=staged,
        episodes=staged.episodes,
        examples_by_id=final_slots,
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
    )
    validated_atoms = validate_development_control_evidence(
        raw_root, inputs=inputs, registry=registry
    )
    if validated_atoms.record_count != 144:
        raise M5AFinalVerificationError("raw final control evidence must contain 144 atoms")

    oracle = _object(
        read_object(root / "oracle_episodes.json", label="Oracle episodes"), label="Oracle"
    )
    learned = _object(
        read_object(root / "neuro_symbolic_episodes.json", label="Neuro episodes"),
        label="Neuro",
    )
    oracle_records = [
        _object(value, label="Oracle episode")
        for value in _sequence(oracle.get("episodes"), label="Oracle episodes")
    ]
    learned_records = [
        _object(value, label="Neuro episode")
        for value in _sequence(learned.get("episodes"), label="Neuro episodes")
    ]
    probes = read_object(root / "rejection_probes.json", label="rejection probes")
    analysis = analyze_final_control_records(
        oracle_records=oracle_records,
        neuro_symbolic_records=learned_records,
        rejection_probe_set=probes,
        examples_by_id=final_slots,
        router_summaries=validated_atoms.summaries,
    )
    stored_control_gate = read_object(root / "control_quality_gate.json", label="control gate")
    _equal(stored_control_gate.get("gate_items"), analysis["gate_items"], label="control gate")
    if (
        stored_control_gate.get("final_control_quality_passed")
        is not analysis["final_control_quality_passed"]
    ):
        raise M5AFinalVerificationError("final control quality flag differs")
    stored_result = read_object(root / "final_result.json", label="final result")
    final_result = build_final_result(
        language_gate=language_gate,
        control_analysis=analysis,
        independent_evidence_valid=True,
    )
    _equal(stored_result, final_result, label="final result")

    flags = cast(Mapping[str, object], cast(Mapping[str, object], evidence["complete"])["flags"])
    required_false = {
        "test_split_accessed",
        "historical_fresh_accessed",
        "m42_final_accessed",
        "smolvla_go",
    }
    if any(flags.get(key) is not False for key in required_false):
        raise M5AFinalVerificationError("a prohibited final source or SmolVLA was accessed")
    expected_flags = {
        "final_pipeline_validated": True,
        "final_quality_gate_passed": final_result["final_quality_gate_passed"] is True,
        "physical_target_validated": True,
    }
    if any(flags.get(key) is not value for key, value in expected_flags.items()):
        raise M5AFinalVerificationError("final completion flags differ")
    base: dict[str, object] = {
        "schema_version": SEALED_FINAL_VERIFICATION_SCHEMA,
        "passed": True,
        "final_authorization_validated": True,
        "sealed_language_final_accessed": True,
        "sealed_control_final_accessed": True,
        "final_language_evaluation_completed": True,
        "final_language_safety_quality_passed": language_gate[
            "final_language_safety_quality_passed"
        ],
        "final_rejection_taxonomy_quality_passed": language_gate[
            "final_rejection_taxonomy_quality_passed"
        ],
        "final_control_schedule_validated": True,
        "final_oracle_control_completed": True,
        "final_neuro_symbolic_control_completed": True,
        "final_paired_states_validated": analysis["paired_initial_state_count"] == 72,
        "final_routing_results_validated": True,
        "final_rejection_no_dispatch_validated": True,
        "final_failure_attribution_validated": True,
        "final_action_runtime_validated": True,
        "final_control_quality_passed": analysis["final_control_quality_passed"],
        "final_pipeline_validated": True,
        "final_quality_gate_passed": final_result["final_quality_gate_passed"],
        "physical_target_validated": True,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "final_access_attempt_count": 1,
        "language_example_count": len(final_examples),
        "control_pair_count": 72,
        "control_atom_count": validated_atoms.record_count,
        "run_fingerprint": reconstructed.run_fingerprint,
        "artifact_fingerprint": evidence["artifact_fingerprint"],
        "language_quality_gate": language_gate,
        "control_analysis": analysis,
        "final_result": final_result,
    }
    return {**base, "verification_fingerprint": f"sha256:{sha256_hex(base)}"}


def _structural() -> dict[str, object]:
    contract = final_metric_contract()
    return {
        "schema_version": SEALED_FINAL_VERIFICATION_SCHEMA,
        "mode": "structural",
        "passed": True,
        "final_schedule_materialized": False,
        "final_access_attempt_count": 0,
        "metric_contract_fingerprint": contract["contract_fingerprint"],
        "physical_target_validated": False,
        "smolvla_go": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report: dict[str, object] = {
        "schema_version": SEALED_FINAL_VERIFICATION_SCHEMA,
        "passed": False,
        "physical_target_validated": False,
        "smolvla_go": False,
    }
    try:
        report = _structural() if args.structural else _verify_target(args)
    except Exception as error:  # noqa: BLE001 - verifier preserves exact failure
        traceback.print_exc()
        report.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error) or repr(error),
                "test_split_accessed": False,
                "historical_fresh_accessed": False,
                "m42_final_accessed": False,
                "smolvla_go": False,
            }
        )
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
