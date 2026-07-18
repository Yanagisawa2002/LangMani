"""M5A.2 language-only candidate roles, gates, and immutable selection contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import torch
import torch.nn.functional as F

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import TaskSpec
from langmani.language.classifier_decoders import (
    DecoderCandidate,
    DecoderConfigurationV0,
    decode_classifier_probabilities,
)
from langmani.language.llm_router import (
    QWEN3_1_7B_FILE_IDENTITIES,
    QWEN3_1_7B_LICENSE,
    QWEN3_1_7B_MODEL_ID,
    QWEN3_1_7B_REVISION,
)
from langmani.language.rejection_diagnostics import ReadOnlyClassifierBundleV0
from langmani.language.router_types import (
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)

M5A2_CONTRACT_SCHEMA = "langmani-m5a2-offline-language-development-v0"
CLASSIFIER_NEGATIVE_ROUTER_NAME = "FactorizedTextClassifierNegativeBaselineV0"
CLASSIFIER_NEGATIVE_ROUTER_VERSION = "factorized-text-classifier-negative-baseline-v0"


class M5A2ContractError(ValueError):
    """Raised when offline language-development evidence violates M5A.2."""


class LanguageCandidateRole(StrEnum):
    RULE_BASELINE = "deterministic_offline_baseline"
    CLASSIFIER_NEGATIVE_BASELINE = "frozen_offline_negative_baseline"
    LOCAL_LLM_CANDIDATE = "learned_language_candidate"


@dataclass(frozen=True, slots=True)
class CandidateEligibilityV0:
    candidate_id: str
    role: LanguageCandidateRole
    offline_evaluation_eligible: bool
    controller_dispatch_eligible: bool
    promotion_eligible: bool
    final_selection_eligible: bool
    deterministic_baseline: bool

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not isinstance(self.role, LanguageCandidateRole):
            raise M5A2ContractError("candidate eligibility requires one stable candidate and role")
        expected_candidate = {
            LanguageCandidateRole.RULE_BASELINE: "RuleRouterV0",
            LanguageCandidateRole.CLASSIFIER_NEGATIVE_BASELINE: (CLASSIFIER_NEGATIVE_ROUTER_NAME),
            LanguageCandidateRole.LOCAL_LLM_CANDIDATE: "StructuredLocalLLMRouterV0",
        }[self.role]
        if self.candidate_id != expected_candidate:
            raise M5A2ContractError("candidate identity cannot be relabeled as another role")
        if self.role is LanguageCandidateRole.CLASSIFIER_NEGATIVE_BASELINE and any(
            (
                self.controller_dispatch_eligible,
                self.promotion_eligible,
                self.final_selection_eligible,
                self.deterministic_baseline,
            )
        ):
            raise M5A2ContractError("the rejected classifier can never dispatch or promote")
        if self.role is LanguageCandidateRole.RULE_BASELINE and (
            not self.deterministic_baseline or self.promotion_eligible
        ):
            raise M5A2ContractError(
                "RuleRouter must remain a non-promotable deterministic baseline"
            )
        if self.role is LanguageCandidateRole.LOCAL_LLM_CANDIDATE and (
            self.deterministic_baseline or not self.promotion_eligible
        ):
            raise M5A2ContractError("the local LLM is the only learned promotion candidate")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "role": self.role.value,
            "offline_evaluation_eligible": self.offline_evaluation_eligible,
            "controller_dispatch_eligible": self.controller_dispatch_eligible,
            "promotion_eligible": self.promotion_eligible,
            "final_selection_eligible": self.final_selection_eligible,
            "deterministic_baseline": self.deterministic_baseline,
        }


RULE_ROUTER_ELIGIBILITY = CandidateEligibilityV0(
    candidate_id="RuleRouterV0",
    role=LanguageCandidateRole.RULE_BASELINE,
    offline_evaluation_eligible=True,
    controller_dispatch_eligible=False,
    promotion_eligible=False,
    final_selection_eligible=False,
    deterministic_baseline=True,
)
CLASSIFIER_NEGATIVE_ELIGIBILITY = CandidateEligibilityV0(
    candidate_id=CLASSIFIER_NEGATIVE_ROUTER_NAME,
    role=LanguageCandidateRole.CLASSIFIER_NEGATIVE_BASELINE,
    offline_evaluation_eligible=True,
    controller_dispatch_eligible=False,
    promotion_eligible=False,
    final_selection_eligible=False,
    deterministic_baseline=False,
)
LOCAL_LLM_ELIGIBILITY = CandidateEligibilityV0(
    candidate_id="StructuredLocalLLMRouterV0",
    role=LanguageCandidateRole.LOCAL_LLM_CANDIDATE,
    offline_evaluation_eligible=True,
    controller_dispatch_eligible=False,
    promotion_eligible=True,
    final_selection_eligible=True,
    deterministic_baseline=False,
)


@dataclass(frozen=True, slots=True)
class QwenModelIdentityV0:
    model_id: str = QWEN3_1_7B_MODEL_ID
    model_revision: str = QWEN3_1_7B_REVISION
    tokenizer_revision: str = QWEN3_1_7B_REVISION
    license: str = QWEN3_1_7B_LICENSE
    model_card_url: str = "https://huggingface.co/Qwen/Qwen3-1.7B"
    official_chat_template: bool = True
    non_thinking_method: str = "tokenizer.apply_chat_template(enable_thinking=False)"

    def __post_init__(self) -> None:
        if (
            self.model_id != QWEN3_1_7B_MODEL_ID
            or self.model_revision != QWEN3_1_7B_REVISION
            or self.tokenizer_revision != QWEN3_1_7B_REVISION
            or self.license != QWEN3_1_7B_LICENSE
            or len(self.model_revision) != 40
        ):
            raise M5A2ContractError("M5A.2 permits exactly one pinned Qwen3-1.7B identity")

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "langmani-m5a2-qwen-model-identity-v0",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "license": self.license,
            "model_card_url": self.model_card_url,
            "official_chat_template": self.official_chat_template,
            "non_thinking_method": self.non_thinking_method,
            "expected_files": {
                name: {"size_bytes": size, "sha256": f"sha256:{digest}"}
                for name, (size, digest) in QWEN3_1_7B_FILE_IDENTITIES.items()
            },
        }


@dataclass(frozen=True, slots=True)
class PromptLockV0:
    prompt_text: str
    prompt_fingerprint: str
    prompt_example_ids: tuple[str, ...]
    train_smoke_example_ids: tuple[str, ...]
    corpus_fingerprint: str
    locked_before_validation: bool

    def __post_init__(self) -> None:
        if not self.prompt_text or len(set(self.prompt_example_ids)) != len(
            self.prompt_example_ids
        ):
            raise M5A2ContractError("prompt lock requires one non-empty prompt and unique examples")
        expected = f"sha256:{sha256_hex({'prompt_template': self.prompt_text, 'example_ids': self.prompt_example_ids})}"
        if self.prompt_fingerprint != expected:
            raise M5A2ContractError("prompt fingerprint differs from its exact text/examples")
        if not self.train_smoke_example_ids or not self.locked_before_validation:
            raise M5A2ContractError("prompt must be locked after train smoke and before validation")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "langmani-m5a2-prompt-lock-v0",
            "prompt_text": self.prompt_text,
            "prompt_fingerprint": self.prompt_fingerprint,
            "prompt_example_ids": list(self.prompt_example_ids),
            "train_smoke_example_ids": list(self.train_smoke_example_ids),
            "corpus_fingerprint": self.corpus_fingerprint,
            "prompt_example_split": "train",
            "train_smoke_split": "train",
            "locked_before_validation": self.locked_before_validation,
            "prompt_sweep_count": 1,
        }


@dataclass(frozen=True, slots=True)
class QualityGateResultV0:
    gate_id: str
    checks: Mapping[str, bool]

    def __post_init__(self) -> None:
        if (
            not self.gate_id.strip()
            or not self.checks
            or not all(
                isinstance(key, str) and isinstance(value, bool)
                for key, value in self.checks.items()
            )
        ):
            raise M5A2ContractError("quality gate requires named boolean checks")

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {"gate_id": self.gate_id, "checks": dict(self.checks), "passed": self.passed}


def _metric(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise M5A2ContractError(f"required finite metric is missing: {key}")
    return float(value)


def evaluate_llm_validation_gate(
    *, metrics: Mapping[str, object], prohibited_source_access: bool
) -> QualityGateResultV0:
    return QualityGateResultV0(
        gate_id="m5a2-llm-validation-gate-v0",
        checks={
            "full_task_spec_accuracy_at_least_0_95": _metric(metrics, "valid_full_task_accuracy")
            >= 0.95,
            "target_object_accuracy_at_least_0_97": _metric(metrics, "object_accuracy") >= 0.97,
            "destination_bin_accuracy_at_least_0_97": _metric(metrics, "bin_accuracy") >= 0.97,
            "false_route_rate_at_most_0_03": _metric(metrics, "false_route_rate") <= 0.03,
            "ambiguous_rejection_recall_at_least_0_90": _metric(
                metrics, "ambiguous_rejection_recall"
            )
            >= 0.90,
            "unsupported_rejection_recall_at_least_0_95": _metric(
                metrics, "unsupported_rejection_recall"
            )
            >= 0.95,
            "malformed_rejection_recall_at_least_0_95": _metric(
                metrics, "malformed_rejection_recall"
            )
            >= 0.95,
            "final_schema_valid_output_rate_at_least_0_99": _metric(
                metrics, "final_schema_valid_output_rate"
            )
            >= 0.99,
            "malformed_after_repair_at_most_0_01": _metric(
                metrics, "malformed_output_after_repair_rate"
            )
            <= 0.01,
            "deterministic_repeatability_exactly_1": _metric(metrics, "deterministic_repeatability")
            == 1.0,
            "no_prohibited_source_access": not prohibited_source_access,
        },
    )


def evaluate_llm_development_gate(
    *,
    metrics: Mapping[str, object],
    per_task_accuracy: Mapping[str, object],
    rejection_family_false_route_rates: Mapping[str, object],
    safety_checks: Mapping[str, bool],
) -> QualityGateResultV0:
    base = dict(
        evaluate_llm_validation_gate(metrics=metrics, prohibited_source_access=False).checks
    )
    base.update(
        {
            "every_task_spec_accuracy_at_least_0_90": bool(per_task_accuracy)
            and all(_metric(per_task_accuracy, key) >= 0.90 for key in per_task_accuracy),
            "no_rejection_family_false_route_rate_above_0_10": all(
                _metric(rejection_family_false_route_rates, key) <= 0.10
                for key in rejection_family_false_route_rates
            ),
            "empty_and_meaningless_inputs_never_route": safety_checks.get(
                "empty_and_meaningless_inputs_never_route"
            )
            is True,
            "conflicting_object_and_bin_inputs_never_route": safety_checks.get(
                "conflicting_object_and_bin_inputs_never_route"
            )
            is True,
            "unsupported_action_commands_never_route": safety_checks.get(
                "unsupported_action_commands_never_route"
            )
            is True,
        }
    )
    return QualityGateResultV0(gate_id="m5a2-llm-development-gate-v0", checks=base)


@dataclass(frozen=True, slots=True)
class LanguageRouterSelectionV0:
    runtime_fingerprint: str
    model_identity_fingerprint: str
    prompt_fingerprint: str
    parser_fingerprint: str
    generation_config_fingerprint: str
    validation_evidence_fingerprint: str
    development_evidence_fingerprint: str
    git_commit: str
    corpus_fingerprint: str
    validation_split_fingerprint: str
    development_split_fingerprint: str
    confidence_available: bool = False

    def __post_init__(self) -> None:
        values = (
            self.runtime_fingerprint,
            self.model_identity_fingerprint,
            self.prompt_fingerprint,
            self.parser_fingerprint,
            self.generation_config_fingerprint,
            self.validation_evidence_fingerprint,
            self.development_evidence_fingerprint,
            self.corpus_fingerprint,
            self.validation_split_fingerprint,
            self.development_split_fingerprint,
        )
        if any(not value.startswith("sha256:") for value in values):
            raise M5A2ContractError("selected learned router must bind all content identities")
        if len(self.git_commit) != 40 or self.confidence_available:
            raise M5A2ContractError("selection requires a full Git commit and no fake confidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "langmani-m5a2-language-router-selection-v0",
            "selected_router": "StructuredLocalLLMRouterV0",
            "runtime_fingerprint": self.runtime_fingerprint,
            "model_identity_fingerprint": self.model_identity_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "parser_fingerprint": self.parser_fingerprint,
            "generation_config_fingerprint": self.generation_config_fingerprint,
            "repair_policy": "at_most_one_schema_repair_v0",
            "validation_evidence_fingerprint": self.validation_evidence_fingerprint,
            "development_evidence_fingerprint": self.development_evidence_fingerprint,
            "git_commit": self.git_commit,
            "corpus_fingerprint": self.corpus_fingerprint,
            "validation_split_fingerprint": self.validation_split_fingerprint,
            "development_split_fingerprint": self.development_split_fingerprint,
            "confidence_available": self.confidence_available,
        }


_CLASSIFIER_REASON_BY_STATUS = {
    RouterStatus.REJECT_AMBIGUOUS: RouterRejectionReason.CLASSIFIER_AMBIGUOUS,
    RouterStatus.REJECT_UNSUPPORTED: RouterRejectionReason.CLASSIFIER_UNSUPPORTED,
    RouterStatus.REJECT_MALFORMED: RouterRejectionReason.CLASSIFIER_MALFORMED,
}


class FrozenClassifierNegativeBaselineV0:
    """Read-only selected M5A.1 decoder; it deliberately has no dispatch interface."""

    def __init__(
        self,
        *,
        bundle: ReadOnlyClassifierBundleV0,
        configuration: DecoderConfigurationV0,
        maximum_sequence_length: int,
        device: str = "cpu",
    ) -> None:
        if configuration.candidate is not DecoderCandidate.CONSERVATIVE_ROUTE_DECODER_V0:
            raise M5A2ContractError("negative baseline requires the frozen conservative decoder")
        if device not in {"cpu", "cuda"} or maximum_sequence_length <= 0:
            raise M5A2ContractError("negative baseline runtime configuration is malformed")
        self.bundle = bundle
        self.configuration = configuration
        self.maximum_sequence_length = maximum_sequence_length
        self.device = device
        self.model = bundle.model.to(device).eval()
        self.tokenizer = bundle.tokenizer
        self.model_state_fingerprint_before = model_state_fingerprint(self.model)

    def route(self, command: str) -> RouterDecision:
        if not isinstance(command, str):
            raise TypeError("command must be text")
        encoded = cast(
            Mapping[str, torch.Tensor],
            self.tokenizer(
                command,
                padding=False,
                truncation=True,
                max_length=self.maximum_sequence_length,
                return_tensors="pt",
            ),
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
            )
        status = F.softmax(
            output.status_logits[0].detach().cpu().double() / self.configuration.status_temperature,
            dim=-1,
        ).tolist()
        objects = F.softmax(output.object_logits[0].detach().cpu().double(), dim=-1).tolist()
        bins = F.softmax(output.bin_logits[0].detach().cpu().double(), dim=-1).tolist()
        decoded = decode_classifier_probabilities(
            status_probabilities=status,
            object_probabilities=objects,
            bin_probabilities=bins,
            configuration=self.configuration,
        )
        evidence = {
            "offline_negative_baseline": True,
            "eligible_for_promotion": False,
            "eligible_for_dispatch": False,
            "configuration": self.configuration.to_dict(),
            "decoder_decision": decoded.to_dict(),
            "status_probabilities": status,
            "object_probabilities": objects,
            "bin_probabilities": bins,
        }
        confidence = RouterConfidence.unavailable(
            definition="rejected classifier probability is descriptive and not a runtime confidence"
        )
        if decoded.status is RouterStatus.ROUTE:
            assert decoded.target_object_id is not None and decoded.target_bin_id is not None
            return RouterDecision.route(
                task_spec=TaskSpec(decoded.target_object_id, decoded.target_bin_id, "canonical_v0"),
                confidence=confidence,
                router_name=CLASSIFIER_NEGATIVE_ROUTER_NAME,
                router_version=CLASSIFIER_NEGATIVE_ROUTER_VERSION,
                evidence=evidence,
            )
        return RouterDecision.reject(
            status=decoded.status,
            rejection_reason=_CLASSIFIER_REASON_BY_STATUS[decoded.status],
            confidence=confidence,
            router_name=CLASSIFIER_NEGATIVE_ROUTER_NAME,
            router_version=CLASSIFIER_NEGATIVE_ROUTER_VERSION,
            evidence=evidence,
        )

    def verify_unchanged(self) -> bool:
        return model_state_fingerprint(self.model) == self.model_state_fingerprint_before


def model_state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.numpy().tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def conservative_decoder_from_mapping(value: Mapping[str, object]) -> DecoderConfigurationV0:
    try:
        return DecoderConfigurationV0(
            candidate=DecoderCandidate(cast(str, value["candidate"])),
            temperature_mode=cast(str, value["temperature_mode"]),
            status_temperature=cast(float, value["status_temperature"]),
            route_threshold=cast(float, value["route_threshold"]),
            route_margin_threshold=cast(float, value["route_margin_threshold"]),
            object_confidence_threshold=cast(float, value["object_confidence_threshold"]),
            bin_confidence_threshold=cast(float, value["bin_confidence_threshold"]),
            configuration_fingerprint=cast(str, value["configuration_fingerprint"]),
            schema_version=cast(str, value["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise M5A2ContractError(
            f"selected negative-baseline decoder is malformed: {error}"
        ) from error


__all__ = [
    "CLASSIFIER_NEGATIVE_ELIGIBILITY",
    "CLASSIFIER_NEGATIVE_ROUTER_NAME",
    "CLASSIFIER_NEGATIVE_ROUTER_VERSION",
    "LOCAL_LLM_ELIGIBILITY",
    "M5A2_CONTRACT_SCHEMA",
    "RULE_ROUTER_ELIGIBILITY",
    "CandidateEligibilityV0",
    "FrozenClassifierNegativeBaselineV0",
    "LanguageCandidateRole",
    "LanguageRouterSelectionV0",
    "M5A2ContractError",
    "PromptLockV0",
    "QualityGateResultV0",
    "QwenModelIdentityV0",
    "conservative_decoder_from_mapping",
    "evaluate_llm_development_gate",
    "evaluate_llm_validation_gate",
    "model_state_fingerprint",
]
