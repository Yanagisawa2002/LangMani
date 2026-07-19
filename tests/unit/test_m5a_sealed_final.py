from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.language.failure_attribution import FailureAttribution
from langmani.language.full_control_development import (
    FINAL_AUTHORIZATION_SCHEMA,
    final_evaluation_protocol_fingerprint,
)
from langmani.language.neuro_symbolic_safety import (
    RejectionTaxonomyEvaluation,
    SafeRejectionRecord,
    SafetyRoutingEvaluation,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
    stable_language_example_id,
)
from langmani.language.sealed_final import (
    FINAL_REJECTION_CASES,
    SEALED_FINAL_EVIDENCE_SCHEMA,
    M5AFinalRunIdentityV0,
    SealedFinalContractError,
    analyze_final_control_records,
    build_final_rejection_probe_set,
    build_final_result,
    evaluate_final_language_quality,
    final_metric_contract,
    validate_final_authorization,
)
from langmani.language.sealed_final_evidence import (
    SealedFinalEvidenceError,
    close_final_attempt,
    final_access_count,
    open_final_attempt,
    validate_sealed_final_evidence,
    write_sealed_final_evidence,
)
from langmani.language.three_scene_control import initial_scene_state_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHA = "sha256:" + "1" * 64


def _command() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_m5a_sealed_final.py"
    spec = importlib.util.spec_from_file_location("langmani_test_sealed_final_command", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity() -> M5AFinalRunIdentityV0:
    return M5AFinalRunIdentityV0(
        implementation_git="a" * 40,
        final_authorization_fingerprint=SHA,
        router_lock_fingerprint=SHA,
        controller_registry_fingerprint=SHA,
        runtime_fingerprint=SHA,
        language_schedule_fingerprint=SHA,
        control_schedule_fingerprint=SHA,
        corpus_fingerprint=SHA,
        split_fingerprints={"train": SHA, "final": SHA},
        model_tokenizer_identity_fingerprint=SHA,
        prompt_fingerprint=SHA,
        semantic_schema_fingerprint=SHA,
        symbolic_parser_fingerprint=SHA,
        safety_arbiter_fingerprint=SHA,
        metric_contract_fingerprint=SHA,
        evidence_schema_fingerprint=SHA,
    )


def _authorization() -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": FINAL_AUTHORIZATION_SCHEMA,
        "authorized": True,
        "locked_router_fingerprint": SHA,
        "controller_registry_fingerprint": SHA,
        "runtime_fingerprint": SHA,
        "final_language_schedule_lock_fingerprint": SHA,
        "final_control_schedule_lock_fingerprint": SHA,
        "final_evaluation_protocol_fingerprint": final_evaluation_protocol_fingerprint(),
        "selected_router_identity": "NeuroSymbolicRouterV0",
        "final_data_accessed": False,
        "final_command_texts_materialized": False,
        "final_scene_seeds_materialized": False,
        "automatic_final_execution": False,
    }
    return {**base, "authorization_fingerprint": f"sha256:{sha256_hex(base)}"}


def _safe_probes() -> dict[str, object]:
    decision = RouterDecision.reject(
        status=RouterStatus.REJECT_UNSUPPORTED,
        rejection_reason=RouterRejectionReason.UNSUPPORTED_ACTION,
        confidence=RouterConfidence.unavailable(),
        router_name="NeuroSymbolicRouterV0",
        router_version="neuro-symbolic-router-v0",
    ).to_dict()
    dispatch = {
        "dispatched": False,
        "controller_loaded": False,
        "policy_called": False,
        "environment_reset_called": False,
        "environment_step_count": 0,
        "safe_rejection": True,
    }
    items = [
        {
            **case.to_dict(),
            "policy_reset_count": 0,
            "probe": {
                "passed": True,
                "controller_lookup_count": 0,
                "environment_reset_count": 0,
                "environment_step_count": 0,
                "result": {"decision": decision, "dispatch": dispatch},
            },
        }
        for case in FINAL_REJECTION_CASES
    ]
    return build_final_rejection_probe_set(items)


def _examples() -> dict[str, LanguageExample]:
    examples: dict[str, LanguageExample] = {}
    for index, task in enumerate(CANONICAL_TASK_SPECS):
        raw = f"fixture final command {index}"
        provenance = {"fixture": "sealed-final", "index": index}
        example_id = stable_language_example_id(
            raw_text=raw,
            normalized_text=raw,
            template_family_id=f"final_fixture_{index}",
            lexical_variant_ids=(f"v{index}",),
            expected_status=RouterStatus.ROUTE,
            expected_task_spec=task,
            expected_rejection_reason=None,
            generation_provenance=provenance,
            split=LanguageSplit.FINAL,
        )
        examples[example_id] = LanguageExample(
            example_id=example_id,
            raw_text=raw,
            normalized_text=raw,
            template_family_id=f"final_fixture_{index}",
            lexical_variant_ids=(f"v{index}",),
            expected_status=RouterStatus.ROUTE,
            expected_task_spec=task,
            expected_rejection_reason=None,
            generation_provenance=provenance,
            split=LanguageSplit.FINAL,
        )
    return examples


def _state(scene: int) -> dict[str, object]:
    offset = scene / 1000.0
    return {
        "object_poses": {
            "red_cube": [0.1 + offset, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0],
            "green_cube": [0.2 + offset, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0],
            "blue_cube": [0.3 + offset, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0],
        },
        "bin_poses": {
            "left_bin": [0.4, 0.2, 0.02, 1.0, 0.0, 0.0, 0.0],
            "right_bin": [0.4, -0.2, 0.02, 1.0, 0.0, 0.0, 0.0],
        },
        "panda_qpos": [0.0] * 9,
    }


def _control_fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, LanguageExample],
    dict[str, object],
]:
    examples = _examples()
    by_task = {value.task_id: value for value in examples.values()}
    oracle: list[dict[str, object]] = []
    learned: list[dict[str, object]] = []
    for index in range(72):
        task = CANONICAL_TASK_SPECS[index % 6]
        task_id = stable_task_id(task)
        example = by_task[task_id]
        scene = index // 6
        identity = {
            "schedule_id": "m5a_final_control_v0",
            "schedule_fingerprint": SHA,
            "episode_index": index,
            "scene_seed": 9000 + scene,
            "scene_id": f"scene-{scene}",
            "oracle_task_id": task_id,
            "language_example_id": example.example_id,
            "command_fingerprint": f"sha256:{sha256_hex(example.raw_text)}",
        }
        state = _state(scene)
        audit = {
            "initial_physical_state": state,
            "initial_physical_state_fingerprint": initial_scene_state_fingerprint(state),
            "rollout": {"action_bound_mode": "project", "target_grasped_any": True},
        }
        dispatch = {
            "controller_task_id": task_id,
            "controller_checkpoint_fingerprint": SHA,
        }
        oracle.append(
            {
                "identity": {**identity, "router_label": "oracle"},
                "result": {
                    "dispatch": copy.deepcopy(dispatch),
                    "end_to_end_success": True,
                    "failure_attribution": (
                        FailureAttribution.ROUTING_CORRECT_CONTROL_SUCCESS.value
                    ),
                },
                "paired_execution_audit": copy.deepcopy(audit),
            }
        )
        learned.append(
            {
                "identity": {**identity, "router_label": "neuro_symbolic"},
                "result": {
                    "decision": RouterDecision.route(
                        task_spec=task,
                        confidence=RouterConfidence.unavailable(),
                        router_name="NeuroSymbolicRouterV0",
                        router_version="neuro-symbolic-router-v0",
                    ).to_dict(),
                    "oracle_task_spec": task.to_dict(),
                    "dispatch": copy.deepcopy(dispatch),
                    "end_to_end_success": True,
                    "failure_attribution": (
                        FailureAttribution.ROUTING_CORRECT_CONTROL_SUCCESS.value
                    ),
                },
                "paired_execution_audit": copy.deepcopy(audit),
            }
        )
    summary = {
        "end_to_end_success_count": 72,
        "target_in_wrong_bin_count": 0,
        "target_off_table_count": 0,
        "arm_projected_component_count": 0,
        "malformed_action_count": 0,
        "nonfinite_action_count": 0,
        "infrastructure_failure_count": 0,
        "m2_expert_call_count": 0,
    }
    return oracle, learned, examples, {"oracle": dict(summary), "neuro_symbolic": dict(summary)}


def _good_safety() -> SafetyRoutingEvaluation:
    safe = tuple(
        SafeRejectionRecord(
            example_id=f"reject-{index}",
            template_family_id="rejection_fixture",
            expected_status=RouterStatus.REJECT_UNSUPPORTED,
            actual_status=RouterStatus.REJECT_UNSUPPORTED,
            rejection_reason=RouterRejectionReason.UNSUPPORTED_ACTION,
            decision_fingerprint=SHA,
        )
        for index in range(240)
    )
    return SafetyRoutingEvaluation(
        example_count=600,
        routeable_count=360,
        rejectable_count=240,
        executable_route_count=360,
        correct_complete_task_count=360,
        safe_rejection_count=240,
        unsafe_false_route_count=0,
        false_rejection_count=0,
        malformed_decision_count=0,
        schema_valid_count=600,
        deterministic_count=600,
        rejected_with_task_spec_count=0,
        safe_rejections=safe,
        unsafe_false_routes=tuple(),
        false_rejection_example_ids=tuple(),
        malformed_example_ids=tuple(),
    )


def _taxonomy(*, exact: bool) -> RejectionTaxonomyEvaluation:
    count = 240 if exact else 239
    statuses = {
        RouterStatus.ROUTE.value: 360,
        RouterStatus.REJECT_AMBIGUOUS.value: 80,
        RouterStatus.REJECT_UNSUPPORTED.value: 80,
        RouterStatus.REJECT_MALFORMED.value: 80,
    }
    return RejectionTaxonomyEvaluation(
        example_count=600,
        rejected_count=240,
        exact_four_way_status_correct=600 if exact else 599,
        exact_rejection_status_correct=count,
        exact_reason_correct=count,
        expected_status_counts=statuses,
        status_true_positive_counts=(statuses if exact else {**statuses, "reject_malformed": 79}),
        taxonomy_confusion_matrix={},
        reason_confusion_matrix={},
        metrics_by_template_family={},
        mismatches=tuple(),
    )


def _metrics() -> dict[str, object]:
    return {
        "valid_full_task_accuracy": 1.0,
        "object_accuracy": 1.0,
        "bin_accuracy": 1.0,
        "per_task_accuracy": {stable_task_id(value): 1.0 for value in CANONICAL_TASK_SPECS},
    }


def _language_gate(*, exact_taxonomy: bool = True) -> dict[str, object]:
    return evaluate_final_language_quality(
        metrics=_metrics(),
        safety=_good_safety(),
        taxonomy=_taxonomy(exact=exact_taxonomy),
        special_route_counts={
            "route_on_unsupported_action": 0,
            "route_on_conflicting_objects": 0,
            "route_on_conflicting_bins": 0,
            "route_on_empty_or_noise_input": 0,
            "route_on_unresolved_correction": 0,
            "route_on_unsupported_spatial_reference": 0,
            "route_on_contradictory_negation": 0,
        },
        rejection_family_unsafe_false_route_rates={"rejection_fixture": 0.0},
        no_prohibited_source_access=True,
    )


def test_final_authorization_is_exact_and_positive() -> None:
    authorization = _authorization()
    fingerprint = str(authorization["authorization_fingerprint"])
    assert (
        validate_final_authorization(
            authorization,
            expected_fingerprint=fingerprint,
            router_fingerprint=SHA,
            controller_registry_fingerprint=SHA,
            runtime_fingerprint=SHA,
            language_schedule_fingerprint=SHA,
            control_schedule_fingerprint=SHA,
        )
        == authorization
    )


def test_final_authorization_rejects_any_identity_change() -> None:
    authorization = _authorization()
    authorization["automatic_final_execution"] = True
    with pytest.raises(SealedFinalContractError):
        validate_final_authorization(
            authorization,
            expected_fingerprint=str(authorization["authorization_fingerprint"]),
            router_fingerprint=SHA,
            controller_registry_fingerprint=SHA,
            runtime_fingerprint=SHA,
            language_schedule_fingerprint=SHA,
            control_schedule_fingerprint=SHA,
        )


def test_final_run_fingerprint_is_stable_and_semantic() -> None:
    identity = _identity()
    assert identity.run_fingerprint == _identity().run_fingerprint
    assert identity.to_dict()["schema_version"] == "langmani-m5a-final-run-identity-v0"


def test_final_metric_contract_has_exact_locked_budgets() -> None:
    contract = final_metric_contract()
    assert contract["control"] == {
        "correct_routes_min": 69,
        "pair_count": 72,
        "wrong_object_routes_max": 3,
        "wrong_bin_routes_max": 3,
        "false_rejections_max": 2,
        "success_gap_max": 4,
        "initial_state_pairs": 72,
    }


def test_single_final_access_attempt_is_enforced(tmp_path: Path) -> None:
    opened = open_final_attempt(tmp_path, run_identity=_identity().to_dict())
    assert final_access_count(tmp_path) == 1
    with pytest.raises(SealedFinalEvidenceError):
        open_final_attempt(tmp_path, run_identity=_identity().to_dict())
    close_final_attempt(
        str(opened["attempt_root"]), completed=False, error={"error_type": "fixture"}
    )


def test_invalid_final_attempt_cannot_be_reopened(tmp_path: Path) -> None:
    opened = open_final_attempt(tmp_path, run_identity=_identity().to_dict())
    close_final_attempt(
        str(opened["attempt_root"]), completed=False, error={"error_type": "fixture"}
    )
    assert final_access_count(tmp_path) == 1
    with pytest.raises(SealedFinalEvidenceError):
        open_final_attempt(tmp_path, run_identity=_identity().to_dict())


def test_sealed_evidence_is_atomic_checksummed_and_immutable(tmp_path: Path) -> None:
    owner = {"run_fingerprint": _identity().run_fingerprint}
    written = write_sealed_final_evidence(
        tmp_path, owner=owner, artifacts={"summary.md": "ok"}, flags={"passed": True}
    )
    validated = validate_sealed_final_evidence(str(written["root"]), expected_owner=owner)
    assert validated["passed"] is True
    with pytest.raises(SealedFinalEvidenceError):
        write_sealed_final_evidence(
            tmp_path, owner=owner, artifacts={"summary.md": "again"}, flags={"passed": True}
        )


def test_sealed_evidence_rejects_unsafe_output_paths(tmp_path: Path) -> None:
    with pytest.raises(SealedFinalEvidenceError):
        write_sealed_final_evidence(
            tmp_path,
            owner={"run_fingerprint": _identity().run_fingerprint},
            artifacts={"../escape.json": {}},
            flags={"passed": True},
        )


def test_final_rejection_set_requires_exact_ten_ordered_safe_probes() -> None:
    probes = _safe_probes()
    assert probes["probe_count"] == 10
    assert probes["controller_lookup_count"] == probes["environment_step_count"] == 0


def test_final_rejection_set_rejects_any_runtime_entry() -> None:
    probes = _safe_probes()
    items = copy.deepcopy(probes["cases"])
    items[0]["probe"]["environment_step_count"] = 1
    with pytest.raises(SealedFinalContractError):
        build_final_rejection_probe_set(items)


def test_final_language_safety_gate_passes_independently() -> None:
    gate = _language_gate()
    assert gate["final_language_safety_quality_passed"] is True


def test_final_taxonomy_gate_is_separate_from_dispatch_safety() -> None:
    gate = _language_gate(exact_taxonomy=False)
    assert gate["final_language_safety_quality_passed"] is True
    assert gate["final_rejection_taxonomy_quality_passed"] is False


def test_final_result_separates_pipeline_completion_from_quality() -> None:
    result = build_final_result(
        language_gate=_language_gate(exact_taxonomy=False),
        control_analysis={"final_control_quality_passed": True},
        independent_evidence_valid=True,
    )
    assert result["final_pipeline_validated"] is True
    assert result["final_quality_gate_passed"] is True
    assert result["final_rejection_taxonomy_quality_passed"] is False


def test_final_control_happy_path_is_exact_72_pairs() -> None:
    oracle, learned, examples, summaries = _control_fixture()
    analysis = analyze_final_control_records(
        oracle_records=oracle,
        neuro_symbolic_records=learned,
        rejection_probe_set=_safe_probes(),
        examples_by_id=examples,
        router_summaries=summaries,
    )
    assert analysis["pair_count"] == 72
    assert analysis["final_control_quality_passed"] is True


def test_final_control_rejects_missing_episode_atom() -> None:
    oracle, learned, examples, summaries = _control_fixture()
    with pytest.raises(SealedFinalContractError):
        analyze_final_control_records(
            oracle_records=oracle[:-1],
            neuro_symbolic_records=learned,
            rejection_probe_set=_safe_probes(),
            examples_by_id=examples,
            router_summaries=summaries,
        )


def test_final_control_requires_twelve_six_task_scene_groups() -> None:
    oracle, learned, examples, summaries = _control_fixture()
    learned[6]["identity"]["scene_id"] = "scene-0"
    oracle[6]["identity"]["scene_id"] = "scene-0"
    with pytest.raises(SealedFinalContractError):
        analyze_final_control_records(
            oracle_records=oracle,
            neuro_symbolic_records=learned,
            rejection_probe_set=_safe_probes(),
            examples_by_id=examples,
            router_summaries=summaries,
        )


def test_final_control_state_mismatch_fails_quality_gate() -> None:
    oracle, learned, examples, summaries = _control_fixture()
    learned[0]["paired_execution_audit"]["initial_physical_state"]["panda_qpos"][0] = 0.1
    analysis = analyze_final_control_records(
        oracle_records=oracle,
        neuro_symbolic_records=learned,
        rejection_probe_set=_safe_probes(),
        examples_by_id=examples,
        router_summaries=summaries,
    )
    assert analysis["gate_items"]["initial_state_pairing_72_of_72"] is False


def test_final_control_controller_mismatch_fails_quality_gate() -> None:
    oracle, learned, examples, summaries = _control_fixture()
    learned[0]["result"]["dispatch"]["controller_checkpoint_fingerprint"] = "sha256:" + "2" * 64
    analysis = analyze_final_control_records(
        oracle_records=oracle,
        neuro_symbolic_records=learned,
        rejection_probe_set=_safe_probes(),
        examples_by_id=examples,
        router_summaries=summaries,
    )
    assert (
        analysis["gate_items"]["controller_identity_agreement_for_correct_routes_complete"] is False
    )


def test_final_control_wrong_object_budget_is_not_rounded() -> None:
    oracle, learned, examples, summaries = _control_fixture()
    for record in learned[:4]:
        record["result"]["decision"]["target_object_id"] = "invalid_cube"
    analysis = analyze_final_control_records(
        oracle_records=oracle,
        neuro_symbolic_records=learned,
        rejection_probe_set=_safe_probes(),
        examples_by_id=examples,
        router_summaries=summaries,
    )
    assert analysis["wrong_object_route_count"] == 4
    assert analysis["final_control_quality_passed"] is False


def test_final_control_success_gap_bound_is_four() -> None:
    oracle, learned, examples, summaries = _control_fixture()
    summaries["neuro_symbolic"]["end_to_end_success_count"] = 67
    analysis = analyze_final_control_records(
        oracle_records=oracle,
        neuro_symbolic_records=learned,
        rejection_probe_set=_safe_probes(),
        examples_by_id=examples,
        router_summaries=summaries,
    )
    assert analysis["absolute_success_gap"] == 5
    assert analysis["final_control_quality_passed"] is False


def test_final_control_malformed_executable_decision_is_hard_failure() -> None:
    oracle, learned, examples, summaries = _control_fixture()
    learned[0]["result"]["decision"]["task_spec"] = None
    analysis = analyze_final_control_records(
        oracle_records=oracle,
        neuro_symbolic_records=learned,
        rejection_probe_set=_safe_probes(),
        examples_by_id=examples,
        router_summaries=summaries,
    )
    assert analysis["malformed_executable_decision_count"] == 1
    assert analysis["final_control_quality_passed"] is False


@pytest.mark.parametrize(
    ("field", "gate"),
    [
        ("target_in_wrong_bin_count", "target_in_wrong_bin_zero"),
        ("target_off_table_count", "target_off_table_zero"),
        ("arm_projected_component_count", "arm_projection_zero"),
        ("malformed_action_count", "malformed_action_zero"),
        ("nonfinite_action_count", "nan_zero"),
        ("m2_expert_call_count", "m2_calls_zero"),
        ("infrastructure_failure_count", "infrastructure_failures_zero"),
    ],
)
def test_final_control_safety_counts_are_hard_gates(field: str, gate: str) -> None:
    oracle, learned, examples, summaries = _control_fixture()
    summaries["neuro_symbolic"][field] = 1
    analysis = analyze_final_control_records(
        oracle_records=oracle,
        neuro_symbolic_records=learned,
        rejection_probe_set=_safe_probes(),
        examples_by_id=examples,
        router_summaries=summaries,
    )
    assert analysis["gate_items"][gate] is False


def test_final_cli_requires_explicit_dry_run_or_target() -> None:
    command = _command()
    with pytest.raises(SystemExit):
        command.parse_args([])


def test_final_cli_dry_run_does_not_authorize_gpu_or_control() -> None:
    command = _command()
    args = command.parse_args(["--dry-run"])
    assert args.dry_run is True and args.target_final is False


def test_final_cli_has_no_model_fallback_or_optimizer() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_m5a_sealed_final.py").read_text(encoding="utf-8")
    assert "Qwen3-8B" not in source
    assert "optimizer" not in source.lower()
    assert "env.step" not in source


def test_final_verifier_structural_mode_does_not_materialize_final() -> None:
    source = (PROJECT_ROOT / "environment" / "verify_m5a_final.py").read_text(encoding="utf-8")
    structural = source[source.index("def _structural") : source.index("def main")]
    assert "materialize_final" not in structural
    assert '"final_schedule_materialized": False' in structural


def test_final_evidence_schema_is_fingerprint_bound() -> None:
    identity = _identity()
    assert identity.evidence_schema_fingerprint == SHA
    assert SEALED_FINAL_EVIDENCE_SCHEMA == "langmani-m5a-sealed-final-evidence-v0"


def test_final_result_never_authorizes_smolvla() -> None:
    result = build_final_result(
        language_gate=_language_gate(),
        control_analysis={"final_control_quality_passed": True},
        independent_evidence_valid=True,
    )
    assert result["smolvla_go"] is False
