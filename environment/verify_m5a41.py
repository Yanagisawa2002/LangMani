"""Verify M5A.4.1 structural contracts or one immutable target evidence root."""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Sequence
from pathlib import Path

from langmani.language.corpus import build_language_corpus
from langmani.language.neuro_symbolic_router import (
    DeterministicSafetyArbiterV0,
    SymbolicLexicalParserV0,
)
from langmani.language.neuro_symbolic_safety import (
    RoutingSafetyClass,
    initial_m5a41_flags,
    rejection_taxonomy_contract,
    safety_contract,
    safety_contract_fingerprint,
    taxonomy_contract_fingerprint,
)
from langmani.language.neuro_symbolic_safety_verifier import verify_m5a41_evidence
from langmani.language.router_types import LanguageSplit
from langmani.policies.act_runtime import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "outputs/diagnostics/m5a/m5a41-verification.json"
SCHEMA = "langmani-m5a41-verification-command-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _structural_verification() -> dict[str, object]:
    corpus = build_language_corpus()
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    flags = initial_m5a41_flags()
    rules = rejection_taxonomy_contract()["rules"]
    checks = {
        "exact_five_safety_classes": tuple(RoutingSafetyClass)
        == (
            RoutingSafetyClass.EXECUTABLE_ROUTE,
            RoutingSafetyClass.SAFE_REJECTION,
            RoutingSafetyClass.UNSAFE_FALSE_ROUTE,
            RoutingSafetyClass.FALSE_REJECTION,
            RoutingSafetyClass.MALFORMED_DECISION,
        ),
        "exact_category_does_not_control_dispatch": safety_contract()[
            "exact_rejection_category_controls_dispatch"
        ]
        is False,
        "taxonomy_is_diagnostic_only": rejection_taxonomy_contract()["diagnostic_only"] is True,
        "taxonomy_has_all_corpus_reasons": isinstance(rules, dict) and len(rules) == 14,
        "arbiter_precedence_unchanged": arbiter.contract_dict()["precedence"]
        == ["malformed", "unsupported", "ambiguous", "route"],
        "symbolic_parser_remains_non_dispatching": parser.contract_dict()["controller_dispatch"]
        is False,
        "development_not_required_for_structural_check": len(
            corpus.manifest.split_manifests[LanguageSplit.DEVELOPMENT].example_ids
        )
        == 420,
        "target_flags_remain_false": not any(
            flags[name]
            for name in (
                "taxonomy_audit_completed",
                "train_smoke_safety_gate_passed",
                "neuro_symbolic_router_locked",
                "language_development_completed",
                "real_gpu_inference_validated",
                "physical_target_validated",
            )
        ),
    }
    return {
        "schema_version": SCHEMA,
        "passed": all(checks.values()),
        "implementation_validated": all(checks.values()),
        "structural_only": True,
        "checks": checks,
        "safety_contract_fingerprint": safety_contract_fingerprint(),
        "taxonomy_contract_fingerprint": taxonomy_contract_fingerprint(),
        "physical_target_validated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict[str, object] = {"schema_version": SCHEMA, "passed": False}
    try:
        report = (
            _structural_verification()
            if args.evidence_root is None
            else verify_m5a41_evidence(args.evidence_root)
        )
    except Exception as error:  # noqa: BLE001 - verifier boundary preserves diagnostics
        traceback.print_exc()
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    atomic_write_json(args.report.resolve(strict=False), report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
