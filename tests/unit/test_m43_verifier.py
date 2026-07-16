from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment import verify_m43


def test_structural_report_requires_only_implementation_flags() -> None:
    report = verify_m43.Report(
        implementation_validated=True,
        semantic_audit_implementation_validated=True,
    )

    assert report.passed is True
    assert report.semantic_audit_completed is False
    assert report.factor_film_training_completed is False
    assert report.final_schedule_accessed is False
    assert report.smolvla_go is False
    assert report.physical_target_validated is False


@pytest.mark.parametrize(
    "field",
    [
        "semantic_audit_completed",
        "factor_film_training_completed",
        "final_benchmark_authorized",
        "final_schedule_accessed",
        "smolvla_go",
        "physical_target_validated",
    ],
)
def test_structural_report_rejects_target_or_future_claims(field: str) -> None:
    report = verify_m43.Report(
        implementation_validated=True,
        semantic_audit_implementation_validated=True,
    )

    setattr(report, field, True)

    assert report.passed is False


def test_verifier_output_rejects_source_overlap() -> None:
    with pytest.raises(RuntimeError, match="must not overlap"):
        verify_m43._validate_output_root(verify_m43.PROJECT_ROOT / "tests" / "m43-report")


def test_structural_checks_cover_cli_math_and_truthful_flags() -> None:
    report = verify_m43.Report()

    verify_m43._structural_checks(report)

    assert report.passed is True
    names = {str(value["name"]) for value in report.checks}
    assert {
        "M4.3a imports",
        "primary distance excludes gripper",
        "range-normalized locked-horizon distance",
        "semantic retrieval fixture",
        "deterministic confusion matrices",
        "portable report serialization",
        "semantic taxonomy contracts",
        "synthetic model reload fixture",
        "complete semantic summary structure",
        "real checkpoint reload remains unclaimed",
        "immutable evidence promotion",
        "immutable evidence corruption rejection",
        "semantic-audit CLI dry-run",
        "test and final access prohibition",
        "truthful physical-validation state",
    } <= names


def test_non_target_main_writes_truthful_verification(tmp_path: Path) -> None:
    assert verify_m43.main(["--output-root", str(tmp_path / "verification")]) == 0

    payload = json.loads(
        (tmp_path / "verification" / "verification.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == verify_m43.REPORT_SCHEMA
    assert payload["verification_mode"] == "non_target_structural"
    assert payload["implementation_validated"] is True
    assert payload["semantic_audit_implementation_validated"] is True
    assert payload["semantic_audit_completed"] is False
    assert payload["state_onehot_semantic_alignment_validated"] is False
    assert payload["tasktoken_semantic_alignment_validated"] is False
    assert payload["factor_film_training_completed"] is False
    assert payload["factor_film_reload_validated"] is False
    assert payload["final_schedule_accessed"] is False
    assert payload["smolvla_go"] is False
    assert payload["physical_target_validated"] is False
    assert payload["passed"] is True
