from __future__ import annotations

import json
from pathlib import Path

import pytest

from langmani.language.text_classifier import load_factorized_text_classifier
from langmani.language.text_training import read_text_classifier_run_evidence
from scripts import train_text_router as cli


@pytest.mark.integration
@pytest.mark.fixture
def test_fixture_command_trains_promotes_and_reloads_without_development_or_final(
    tmp_path: Path,
) -> None:
    report = tmp_path / "fixture-report.json"

    return_code = cli.main(
        [
            "--fixture",
            "--output-root",
            str(tmp_path / "models"),
            "--report",
            str(report),
        ]
    )

    assert return_code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["classifier_fixture_completed"] is True
    assert payload["classifier_training_completed"] is False
    assert payload["classifier_checkpoint_selected"] is True
    assert payload["classifier_calibration_validated"] is True
    assert payload["artifact_reload_validated"] is True
    assert payload["development_accessed"] is False
    assert payload["language_final_accessed"] is False
    assert payload["physical_target_validated"] is False
    artifact = Path(payload["artifact_root"])
    model, tokenizer, manifest = load_factorized_text_classifier(artifact)
    evidence = read_text_classifier_run_evidence(artifact)
    assert model.hidden_size == manifest["hidden_size"]
    assert callable(tokenizer)
    assert evidence["mode"] == "fixture"
    assert evidence["data_usage"]["development_examples_materialized"] is False
    assert evidence["data_usage"]["final_examples_materialized"] is False
    assert evidence["router_config"]["device_independent"] is True
    assert evidence["calibration_selection"] == payload["calibration_selection"]
