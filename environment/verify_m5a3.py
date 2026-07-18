"""Verify M5A.3 contracts or one real Qwen3-4B offline evidence root."""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Sequence
from pathlib import Path

from langmani.language.corpus import build_language_corpus
from langmani.language.llm_router import (
    QWEN3_4B_INSTRUCT_MODEL_ID,
    QWEN3_4B_INSTRUCT_REVISION,
    StructuredLLMRouterConfig,
)
from langmani.language.qwen_scale_escalation import (
    M5A3_AUTHORIZED_CANDIDATES,
    FrozenQwen17BBaselineV0,
    PromptEquivalenceV0,
    Qwen4BInstructIdentityV0,
    initial_m5a3_flags,
    repair_policy_fingerprint,
    router_parser_fingerprint,
    router_schema_fingerprint,
)
from langmani.language.qwen_scale_escalation_verifier import verify_qwen_scale_evidence
from langmani.policies.act_runtime import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "outputs/diagnostics/m5a/qwen4b-escalation-verification.json"
DEFAULT_CLASSIFIER_CHECKPOINT = PROJECT_ROOT / (
    "outputs/models/text-router/authoritative-runs/"
    "9e3ac659fa2b695c843650df35e3779741d94b3dd70b2aec52a429bc4b2edf49/"
    "checkpoints/validation_best.pt"
)
DEFAULT_QWEN17B_EVIDENCE = PROJECT_ROOT / (
    "outputs/diagnostics/m5a/language-development/"
    "291621edfc4218d80ea2184e58fde2ec29d5cec2fc23f9aebaa0e30732406c6b"
)
SCHEMA = "langmani-m5a3-verification-command-v0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--qwen17b-evidence-root", type=Path, default=DEFAULT_QWEN17B_EVIDENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _structural_verification() -> dict[str, object]:
    identity = Qwen4BInstructIdentityV0()
    frozen = FrozenQwen17BBaselineV0()
    config = StructuredLLMRouterConfig(
        model_id=QWEN3_4B_INSTRUCT_MODEL_ID,
        model_revision=QWEN3_4B_INSTRUCT_REVISION,
        tokenizer_revision=QWEN3_4B_INSTRUCT_REVISION,
        chat_template_mode="official_qwen_instruct_non_thinking",
    )
    semantic = "sha256:" + "0" * 64
    rendered = "sha256:" + "1" * 64
    prompt = PromptEquivalenceV0(
        qwen17b_prompt_fingerprint=semantic,
        qwen4b_rendered_prompt_fingerprint=rendered,
        semantic_prompt_content_fingerprint=semantic,
        qwen17b_prompt_example_ids=("fixture-example",),
        qwen4b_prompt_example_ids=("fixture-example",),
        schema_fingerprint=router_schema_fingerprint(),
        parser_fingerprint=router_parser_fingerprint(),
        repair_policy_fingerprint=repair_policy_fingerprint(),
        semantic_prompt_content_equal=True,
    )
    flags = initial_m5a3_flags()
    checks = {
        "exactly_one_escalation_candidate": M5A3_AUTHORIZED_CANDIDATES
        == (QWEN3_4B_INSTRUCT_MODEL_ID,),
        "exact_immutable_revision": identity.model_revision == QWEN3_4B_INSTRUCT_REVISION,
        "apache_2_license": identity.license == "apache-2.0",
        "instruct_checkpoint": identity.instruct_checkpoint,
        "no_quantization": config.quantization == "none",
        "one_repair_maximum": config.maximum_format_repair_attempts == 1,
        "official_non_thinking_transport": config.chat_template_mode
        == "official_qwen_instruct_non_thinking",
        "prompt_semantics_equal": prompt.semantic_prompt_content_equal,
        "qwen17b_frozen": frozen.candidate_frozen and not frozen.promotion_eligible,
        "no_target_claims": not any(
            flags[name]
            for name in (
                "qwen4b_model_identity_validated",
                "qwen4b_train_smoke_completed",
                "qwen4b_validation_completed",
                "qwen4b_language_development_completed",
                "real_gpu_inference_validated",
                "physical_target_validated",
            )
        ),
    }
    return {
        "schema_version": SCHEMA,
        **flags,
        "passed": all(checks.values()),
        "implementation_validated": all(checks.values()),
        "structural_only": True,
        "checks": checks,
        "model_identity_fingerprint": identity.fingerprint,
        "prompt_equivalence_fixture_fingerprint": prompt.fingerprint,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload: dict[str, object] = {
        "schema_version": SCHEMA,
        "passed": False,
        "physical_target_validated": False,
    }
    try:
        if args.evidence_root is None:
            payload = _structural_verification()
        else:
            payload = verify_qwen_scale_evidence(
                args.evidence_root,
                corpus=build_language_corpus(),
                classifier_checkpoint=args.classifier_checkpoint,
                qwen17b_evidence_root=args.qwen17b_evidence_root,
            )
    except Exception as error:  # noqa: BLE001 - preserve independent verifier diagnostics
        traceback.print_exc()
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.report, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if payload.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
