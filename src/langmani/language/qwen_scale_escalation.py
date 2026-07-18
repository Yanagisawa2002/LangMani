"""M5A.3 contracts for one bounded Qwen model-capacity comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from langmani.datasets.identity import sha256_hex
from langmani.language.llm_router import (
    QWEN3_1_7B_MODEL_ID,
    QWEN3_1_7B_REVISION,
    QWEN3_4B_INSTRUCT_FILE_IDENTITIES,
    QWEN3_4B_INSTRUCT_LICENSE,
    QWEN3_4B_INSTRUCT_MODEL_ID,
    QWEN3_4B_INSTRUCT_REVISION,
)
from langmani.language.offline_language_development import (
    QualityGateResultV0,
    evaluate_llm_development_gate,
    evaluate_llm_validation_gate,
)
from langmani.language.schema_validation import (
    ROUTER_OUTPUT_FIELDS,
    ROUTER_OUTPUT_STATUSES,
    ROUTER_REASON_CODE_VERSION,
    ROUTER_REASON_CODES_BY_STATUS,
)

M5A3_CONTRACT_SCHEMA = "langmani-m5a3-qwen-scale-escalation-v0"
M5A3_REPAIR_POLICY = "at_most_one_schema_repair_v0"
M5A3_AUTHORIZED_CANDIDATES = (QWEN3_4B_INSTRUCT_MODEL_ID,)


class M5A3ContractError(ValueError):
    """Raised when the bounded scale-escalation contract is violated."""


@dataclass(frozen=True, slots=True)
class FrozenQwen17BBaselineV0:
    """The completed Qwen3-1.7B result is descriptive and never promotable."""

    model_id: str = QWEN3_1_7B_MODEL_ID
    model_revision: str = QWEN3_1_7B_REVISION
    candidate_frozen: bool = True
    offline_baseline_validated: bool = True
    controller_dispatch_eligible: bool = False
    promotion_eligible: bool = False
    additional_prompt_edit_authorized: bool = False
    additional_repair_authorized: bool = False

    def __post_init__(self) -> None:
        if (
            self.model_id != QWEN3_1_7B_MODEL_ID
            or self.model_revision != QWEN3_1_7B_REVISION
            or not self.candidate_frozen
            or not self.offline_baseline_validated
            or any(
                (
                    self.controller_dispatch_eligible,
                    self.promotion_eligible,
                    self.additional_prompt_edit_authorized,
                    self.additional_repair_authorized,
                )
            )
        ):
            raise M5A3ContractError("Qwen3-1.7B must remain one frozen negative baseline")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "langmani-m5a3-qwen17b-negative-baseline-v0",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "candidate_frozen": self.candidate_frozen,
            "offline_baseline_validated": self.offline_baseline_validated,
            "controller_dispatch_eligible": self.controller_dispatch_eligible,
            "promotion_eligible": self.promotion_eligible,
            "additional_prompt_edit_authorized": self.additional_prompt_edit_authorized,
            "additional_repair_authorized": self.additional_repair_authorized,
        }


@dataclass(frozen=True, slots=True)
class Qwen4BInstructIdentityV0:
    """Exact immutable identity of the sole authorized M5A.3 candidate."""

    model_id: str = QWEN3_4B_INSTRUCT_MODEL_ID
    model_revision: str = QWEN3_4B_INSTRUCT_REVISION
    tokenizer_revision: str = QWEN3_4B_INSTRUCT_REVISION
    license: str = QWEN3_4B_INSTRUCT_LICENSE
    model_card_url: str = "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507"
    official_chat_template: bool = True
    instruct_checkpoint: bool = True
    non_thinking_method: str = "official_instruct_template_without_thinking_switch"

    def __post_init__(self) -> None:
        if (
            M5A3_AUTHORIZED_CANDIDATES != (QWEN3_4B_INSTRUCT_MODEL_ID,)
            or self.model_id != QWEN3_4B_INSTRUCT_MODEL_ID
            or self.model_revision != QWEN3_4B_INSTRUCT_REVISION
            or self.tokenizer_revision != QWEN3_4B_INSTRUCT_REVISION
            or self.license != "apache-2.0"
            or len(self.model_revision) != 40
            or not self.official_chat_template
            or not self.instruct_checkpoint
        ):
            raise M5A3ContractError("M5A.3 permits exactly one pinned Qwen3-4B instruct identity")

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "langmani-m5a3-qwen4b-model-identity-v0",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "license": self.license,
            "model_card_url": self.model_card_url,
            "official_chat_template": self.official_chat_template,
            "instruct_checkpoint": self.instruct_checkpoint,
            "non_thinking_method": self.non_thinking_method,
            "expected_files": {
                name: {"size_bytes": size, "sha256": f"sha256:{digest}"}
                for name, (size, digest) in QWEN3_4B_INSTRUCT_FILE_IDENTITIES.items()
            },
        }


def router_schema_fingerprint() -> str:
    payload = {
        "fields": sorted(ROUTER_OUTPUT_FIELDS),
        "statuses": list(ROUTER_OUTPUT_STATUSES),
        "reason_code_version": ROUTER_REASON_CODE_VERSION,
        "reason_codes_by_status": {
            key: sorted(value) for key, value in sorted(ROUTER_REASON_CODES_BY_STATUS.items())
        },
    }
    return f"sha256:{sha256_hex(payload)}"


def router_parser_fingerprint() -> str:
    return f"sha256:{sha256_hex({'parser': 'parse_strict_router_json', 'schema': router_schema_fingerprint(), 'mixed_route_reject_forbidden': True, 'trailing_text_forbidden': True, 'duplicate_keys_forbidden': True})}"


def repair_policy_fingerprint() -> str:
    return f"sha256:{sha256_hex({'policy': M5A3_REPAIR_POLICY, 'maximum_attempts': 1, 'expected_label_visible': False, 'expected_task_spec_visible': False, 'template_family_visible': False})}"


@dataclass(frozen=True, slots=True)
class PromptEquivalenceV0:
    """Separate frozen semantic content from model-specific transport formatting."""

    qwen17b_prompt_fingerprint: str
    qwen4b_rendered_prompt_fingerprint: str
    semantic_prompt_content_fingerprint: str
    qwen17b_prompt_example_ids: tuple[str, ...]
    qwen4b_prompt_example_ids: tuple[str, ...]
    schema_fingerprint: str
    parser_fingerprint: str
    repair_policy_fingerprint: str
    semantic_prompt_content_equal: bool

    def __post_init__(self) -> None:
        fingerprints = (
            self.qwen17b_prompt_fingerprint,
            self.qwen4b_rendered_prompt_fingerprint,
            self.semantic_prompt_content_fingerprint,
            self.schema_fingerprint,
            self.parser_fingerprint,
            self.repair_policy_fingerprint,
        )
        if any(not value.startswith("sha256:") or len(value) != 71 for value in fingerprints):
            raise M5A3ContractError("prompt equivalence requires complete SHA-256 identities")
        if (
            self.qwen17b_prompt_fingerprint != self.semantic_prompt_content_fingerprint
            or self.qwen17b_prompt_example_ids != self.qwen4b_prompt_example_ids
            or not self.semantic_prompt_content_equal
            or len(set(self.qwen4b_prompt_example_ids)) != len(self.qwen4b_prompt_example_ids)
        ):
            raise M5A3ContractError("M5A.3 prompt semantics or few-shot identities differ")

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "langmani-m5a3-prompt-equivalence-v0",
            "qwen17b_prompt_fingerprint": self.qwen17b_prompt_fingerprint,
            "qwen4b_rendered_prompt_fingerprint": self.qwen4b_rendered_prompt_fingerprint,
            "semantic_prompt_content_fingerprint": self.semantic_prompt_content_fingerprint,
            "qwen17b_prompt_example_ids": list(self.qwen17b_prompt_example_ids),
            "qwen4b_prompt_example_ids": list(self.qwen4b_prompt_example_ids),
            "same_few_shot_ids": self.qwen17b_prompt_example_ids == self.qwen4b_prompt_example_ids,
            "schema_fingerprint": self.schema_fingerprint,
            "parser_fingerprint": self.parser_fingerprint,
            "repair_policy_fingerprint": self.repair_policy_fingerprint,
            "semantic_prompt_content_equal": self.semantic_prompt_content_equal,
            "transport_difference_only": True,
        }


def evaluate_qwen4b_validation_gate(
    *, metrics: Mapping[str, object], prohibited_source_access: bool
) -> QualityGateResultV0:
    """Apply the unchanged M5A.2 thresholds to the sole 4B candidate."""

    source = evaluate_llm_validation_gate(
        metrics=metrics, prohibited_source_access=prohibited_source_access
    )
    return QualityGateResultV0(gate_id="m5a3-qwen4b-validation-gate-v0", checks=source.checks)


def evaluate_qwen4b_development_gate(
    *,
    metrics: Mapping[str, object],
    per_task_accuracy: Mapping[str, object],
    rejection_family_false_route_rates: Mapping[str, object],
    safety_checks: Mapping[str, bool],
) -> QualityGateResultV0:
    source = evaluate_llm_development_gate(
        metrics=metrics,
        per_task_accuracy=per_task_accuracy,
        rejection_family_false_route_rates=rejection_family_false_route_rates,
        safety_checks=safety_checks,
    )
    return QualityGateResultV0(gate_id="m5a3-qwen4b-development-gate-v0", checks=source.checks)


_COMPARISON_KEYS: tuple[str, ...] = (
    "valid_full_task_accuracy",
    "object_accuracy",
    "bin_accuracy",
    "false_route_rate",
    "ambiguous_rejection_recall",
    "unsupported_rejection_recall",
    "malformed_rejection_recall",
    "final_schema_valid_output_rate",
    "repair_attempt_rate",
    "malformed_output_after_repair_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "total_elapsed_inference_seconds",
    "generated_token_count",
    "peak_gpu_memory_bytes",
)


def _finite_metric(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise M5A3ContractError(f"scale comparison lacks finite metric {key}")
    return float(value)


def compute_scale_comparison(
    *,
    qwen17b_metrics: Mapping[str, object],
    qwen4b_metrics: Mapping[str, object],
    validation_example_ids_17b: Sequence[str],
    validation_example_ids_4b: Sequence[str],
    qwen4b_gate_passed: bool,
    model_download_seconds_17b: float | None = None,
    model_download_seconds_4b: float | None = None,
) -> dict[str, object]:
    """Compare capacity only after proving the validation records are identical."""

    ids17 = tuple(validation_example_ids_17b)
    ids4 = tuple(validation_example_ids_4b)
    if not ids17 or ids17 != ids4 or len(set(ids17)) != len(ids17):
        raise M5A3ContractError("scale comparison requires identical ordered validation records")
    deltas = {
        key: _finite_metric(qwen4b_metrics, key) - _finite_metric(qwen17b_metrics, key)
        for key in _COMPARISON_KEYS
    }
    semantic_improved = (
        deltas["valid_full_task_accuracy"] > 0
        or deltas["object_accuracy"] > 0
        or deltas["bin_accuracy"] > 0
    ) and deltas["false_route_rate"] <= 0
    structured_improved = (
        deltas["final_schema_valid_output_rate"] > 0
        and deltas["malformed_output_after_repair_rate"] < 0
    )
    rejection_delta = max(
        deltas["ambiguous_rejection_recall"],
        deltas["unsupported_rejection_recall"],
        deltas["malformed_rejection_recall"],
    )
    return {
        "schema_version": "langmani-m5a3-scale-comparison-v0",
        "validation_example_count": len(ids17),
        "validation_example_ids_fingerprint": f"sha256:{sha256_hex(ids17)}",
        "same_validation_records": True,
        "prompt_semantics_equal": True,
        "delta_definition": "qwen4b_minus_qwen17b",
        "metric_deltas": deltas,
        "model_download_seconds": {
            "qwen17b": model_download_seconds_17b,
            "qwen4b": model_download_seconds_4b,
        },
        "scaling_improved_semantic_routing": semantic_improved,
        "scaling_improved_structured_reliability": structured_improved,
        "rejection_remains_dominant_failure": (
            rejection_delta <= 0
            or min(
                _finite_metric(qwen4b_metrics, "ambiguous_rejection_recall"),
                _finite_metric(qwen4b_metrics, "unsupported_rejection_recall"),
                _finite_metric(qwen4b_metrics, "malformed_rejection_recall"),
            )
            < 0.95
        ),
        "model_reached_promotion_quality": qwen4b_gate_passed,
    }


def initial_m5a3_flags() -> dict[str, bool]:
    return {
        "qwen17b_candidate_frozen": True,
        "qwen17b_offline_baseline_validated": False,
        "qwen4b_model_identity_validated": False,
        "qwen4b_prompt_equivalence_validated": False,
        "qwen4b_train_smoke_completed": False,
        "qwen4b_validation_completed": False,
        "qwen4b_validation_gate_passed": False,
        "qwen4b_language_development_completed": False,
        "qwen4b_language_quality_gate_passed": False,
        "learned_router_selected": False,
        "one_scene_control_smoke_authorized": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "real_gpu_inference_validated": False,
        "physical_target_validated": False,
        "passed": True,
    }


__all__ = [
    "M5A3_AUTHORIZED_CANDIDATES",
    "M5A3_CONTRACT_SCHEMA",
    "M5A3_REPAIR_POLICY",
    "FrozenQwen17BBaselineV0",
    "M5A3ContractError",
    "PromptEquivalenceV0",
    "Qwen4BInstructIdentityV0",
    "compute_scale_comparison",
    "evaluate_qwen4b_development_gate",
    "evaluate_qwen4b_validation_gate",
    "initial_m5a3_flags",
    "repair_policy_fingerprint",
    "router_parser_fingerprint",
    "router_schema_fingerprint",
]
