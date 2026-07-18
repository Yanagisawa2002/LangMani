from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType

import numpy as np
import pytest

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import EpisodeSpec, TaskSpec, stable_task_id
from langmani.language.artifact_validation import (
    CONTROL_ROUTER_ORDER,
    CORPUS_ARCHIVE_SCHEMA,
    ROUTER_EVALUATION_EVIDENCE_SCHEMA,
    M5AArtifactValidationError,
    validate_development_control_evidence,
    validate_language_corpus_archive,
    validate_router_evaluation_evidence,
)
from langmani.language.controller_registry import build_fixture_controller_registry
from langmani.language.corpus import build_language_corpus
from langmani.language.dispatcher import (
    ControllerDispatcher,
    FixturePerTaskControllerLoader,
)
from langmani.language.failure_attribution import (
    FailureAttributionEvidence,
    attribute_end_to_end,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterV0
from langmani.language.schedules import (
    REQUIRED_EXCLUSION_SOURCE_IDS,
    ControlScheduleConfig,
    SeedExclusionSource,
    build_m5a_schedule_bundle,
)
from langmani.language.text_training import (
    TEXT_TRAINING_ARTIFACT_SCHEMA,
    TextTrainingError,
    text_classifier_run_fingerprint,
    validate_completed_text_classifier_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_control_command() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_language_control.py"
    spec = importlib.util.spec_from_file_location("langmani_test_artifact_validation_control", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_language_command() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "evaluate_language_routers.py"
    spec = importlib.util.spec_from_file_location(
        "langmani_test_artifact_validation_language", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _records(root: Path) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "path": PurePosixPath(path.relative_to(root)).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in root.rglob("*")
            if path.is_file() and path.name not in {"artifact_manifest.json", "complete.json"}
        ),
        key=lambda value: str(value["path"]),
    )


def _refresh_router_checksums(root: Path, owner: Mapping[str, object]) -> str:
    artifacts = _records(root)
    artifact_fingerprint = f"sha256:{sha256_hex({'owner': dict(owner), 'artifacts': artifacts})}"
    _write(
        root / "artifact_manifest.json",
        {
            "schema_version": ROUTER_EVALUATION_EVIDENCE_SCHEMA,
            "artifact_fingerprint": artifact_fingerprint,
            "artifacts": artifacts,
        },
    )
    _write(
        root / "complete.json",
        {
            "schema_version": ROUTER_EVALUATION_EVIDENCE_SCHEMA,
            "evaluation_fingerprint": owner["evaluation_fingerprint"],
            "artifact_fingerprint": artifact_fingerprint,
            "passed": True,
        },
    )
    return artifact_fingerprint


def test_corpus_archive_validator_rehashes_every_payload(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    corpus_fingerprint = f"sha256:{'c' * 64}"
    _write(root / "corpus_manifest.json", {"corpus_fingerprint": corpus_fingerprint})
    (root / "train.jsonl").write_text('{"example_id":"one"}\n', encoding="utf-8")
    artifacts = _records(root)
    archive_fingerprint = (
        f"sha256:{sha256_hex({'corpus_fingerprint': corpus_fingerprint, 'artifacts': artifacts})}"
    )
    _write(
        root / "artifact_manifest.json",
        {
            "schema_version": CORPUS_ARCHIVE_SCHEMA,
            "corpus_fingerprint": corpus_fingerprint,
            "archive_fingerprint": archive_fingerprint,
            "artifacts": artifacts,
        },
    )
    _write(
        root / "complete.json",
        {
            "schema_version": CORPUS_ARCHIVE_SCHEMA,
            "corpus_fingerprint": corpus_fingerprint,
            "archive_fingerprint": archive_fingerprint,
            "passed": True,
        },
    )

    validated = validate_language_corpus_archive(
        root, expected_corpus_fingerprint=corpus_fingerprint
    )
    assert validated.archive_fingerprint == archive_fingerprint
    assert validated.artifact_count == 2

    (root / "train.jsonl").write_text('{"example_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(M5AArtifactValidationError, match="manifest differs"):
        validate_language_corpus_archive(root)


def test_classifier_validator_requires_promoted_path_and_rehashes_files(
    tmp_path: Path,
) -> None:
    evidence = {"mode": "fixture", "corpus_fingerprint": f"sha256:{'c' * 64}"}
    classifier_config = {
        "model_id": "fixture/encoder",
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
    }
    run_fingerprint = text_classifier_run_fingerprint(
        classifier_manifest=classifier_config,
        run_evidence=evidence,
    )
    owner = {
        "schema_version": TEXT_TRAINING_ARTIFACT_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "classifier_manifest_fingerprint": f"sha256:{sha256_hex(classifier_config)}",
        "run_evidence_fingerprint": f"sha256:{sha256_hex(evidence)}",
    }
    root = tmp_path / run_fingerprint.removeprefix("sha256:")
    root.mkdir()
    _write(root / "owner.json", owner)
    _write(root / "run_evidence.json", evidence)
    _write(root / "classifier_config.json", classifier_config)
    _write(root / "encoder" / "config.json", {"hidden_size": 8})
    _write(root / "tokenizer" / "tokenizer_config.json", {"model_max_length": 64})
    (root / "router_heads.safetensors").write_bytes(b"fixture-heads")
    artifacts = _records(root)
    artifact_fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': artifacts})}"
    _write(
        root / "artifact_manifest.json",
        {
            "schema_version": TEXT_TRAINING_ARTIFACT_SCHEMA,
            "artifact_fingerprint": artifact_fingerprint,
            "artifacts": artifacts,
        },
    )
    _write(
        root / "complete.json",
        {
            "schema_version": TEXT_TRAINING_ARTIFACT_SCHEMA,
            "artifact_fingerprint": artifact_fingerprint,
            "passed": True,
        },
    )

    assert validate_completed_text_classifier_artifact(root) == root
    (root / "router_heads.safetensors").write_bytes(b"tampered")
    with pytest.raises(TextTrainingError, match="checksum manifest differs"):
        validate_completed_text_classifier_artifact(root)


def test_router_evaluation_validator_rehashes_owner_result_and_payloads(
    tmp_path: Path,
) -> None:
    command = _load_language_command()
    corpus = build_language_corpus()
    examples_by_split: dict[LanguageSplit, tuple[LanguageExample, ...]] = {}
    for split in (LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT):
        available = corpus.examples_for_split(split)
        routeable = next(
            example for example in available if example.expected_status is RouterStatus.ROUTE
        )
        rejected = next(
            example for example in available if example.expected_status is not RouterStatus.ROUTE
        )
        examples_by_split[split] = tuple(
            sorted((routeable, rejected), key=lambda example: example.example_id)
        )

    class PerfectRouter:
        def __init__(self, name: str, examples: tuple[LanguageExample, ...]) -> None:
            self.name = name
            self.examples = {example.raw_text: example for example in examples}

        def route(self, text: str) -> RouterDecision:
            example = self.examples[text]
            evidence = {"parse_errors": []} if self.name == "StructuredLocalLLMRouterV0" else {}
            if example.expected_status is RouterStatus.ROUTE:
                assert example.expected_task_spec is not None
                return RouterDecision.route(
                    task_spec=example.expected_task_spec,
                    confidence=RouterConfidence.unavailable(),
                    router_name=self.name,
                    router_version="fixture-v0",
                    evidence=evidence,
                )
            assert example.expected_rejection_reason is not None
            return RouterDecision.reject(
                status=example.expected_status,
                rejection_reason=example.expected_rejection_reason,
                confidence=RouterConfidence.unavailable(),
                router_name=self.name,
                router_version="fixture-v0",
                evidence=evidence,
            )

    router_names = {
        "rule": "RuleRouterV0",
        "classifier": "FactorizedTextClassifierV0",
        "llm": "StructuredLocalLLMRouterV0",
    }
    owner_payload: dict[str, object] = {
        "schema_version": ROUTER_EVALUATION_EVIDENCE_SCHEMA,
        "mode": "fixture",
        "corpus_fingerprint": f"sha256:{'c' * 64}",
        "corpus_archive_fingerprint": f"sha256:{'a' * 64}",
        "language_validation_fingerprint": f"sha256:{'v' * 64}",
        "language_development_schedule_fingerprint": f"sha256:{'d' * 64}",
        "language_final_lock_fingerprint": f"sha256:{'f' * 64}",
        "router_identities": {
            "rule": {
                "router_name": router_names["rule"],
                "router_fingerprint": f"sha256:{'r' * 64}",
            },
            "classifier": {
                "router_fingerprint": f"sha256:{'c' * 64}",
            },
            "llm": {
                "router_name": router_names["llm"],
                "router_fingerprint": f"sha256:{'l' * 64}",
            },
        },
        "repeat_count": 2,
        "local_files_only": True,
        "evaluation_order": ["validation", "development"],
    }
    evaluation_fingerprint = f"sha256:{sha256_hex(owner_payload)}"
    owner = {**owner_payload, "evaluation_fingerprint": evaluation_fingerprint}
    root = tmp_path / evaluation_fingerprint.removeprefix("sha256:")
    root.mkdir()
    result = {
        "schema_version": ROUTER_EVALUATION_EVIDENCE_SCHEMA,
        "passed": True,
        "mode": owner["mode"],
        "evaluation_fingerprint": evaluation_fingerprint,
        "corpus_fingerprint": owner["corpus_fingerprint"],
        "corpus_archive_fingerprint": owner["corpus_archive_fingerprint"],
        "router_identities": owner["router_identities"],
        "repeat_count": owner["repeat_count"],
        "local_files_only": owner["local_files_only"],
        "evaluation_order": ["validation", "development"],
        "language_validation_schedule": {
            "split": "validation",
            "content_fingerprint": owner["language_validation_fingerprint"],
            "example_count": 2,
        },
        "language_development_schedule": {
            "split": "development",
            "ordered_example_ids": [
                example.example_id for example in examples_by_split[LanguageSplit.DEVELOPMENT]
            ],
            "schedule_fingerprint": owner["language_development_schedule_fingerprint"],
        },
        "language_final_lock": {"sealed": True, "texts_materialized": False},
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "controller_dispatched": False,
        "environment_reset_called": False,
        "environment_step_count": 0,
        "m2_expert_call_count": 0,
        "smolvla_go": False,
    }
    _write(root / "owner.json", owner)
    comparison: dict[str, object] = {}
    for split in (LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT):
        split_comparison: dict[str, object] = {}
        for router_label, router_name in router_names.items():
            records, summary = command._evaluation_payload(
                router=PerfectRouter(router_name, examples_by_split[split]),
                examples=examples_by_split[split],
                split=split,
                repeat_count=2,
                structured_llm=router_label == "llm",
            )
            records_path = root / split.value / router_label / "records.jsonl"
            records_path.parent.mkdir(parents=True, exist_ok=True)
            records_path.write_text(
                "".join(
                    json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            summary_payload = {
                "summary": summary["summary"],
                "supplemental_metrics": summary["supplemental_metrics"],
            }
            _write(root / split.value / router_label / "summary.json", summary_payload)
            split_comparison[router_label] = summary_payload
        comparison[split.value] = split_comparison
    result["comparison"] = comparison
    _write(root / "result.json", result)
    artifact_fingerprint = _refresh_router_checksums(root, owner)

    validated = validate_router_evaluation_evidence(
        root,
        expected_examples_by_split=examples_by_split,
    )
    assert validated.evaluation_fingerprint == evaluation_fingerprint
    assert validated.result == result
    assert validated.artifact_fingerprint == artifact_fingerprint

    result["local_files_only"] = False
    _write(root / "result.json", result)
    _refresh_router_checksums(root, owner)
    with pytest.raises(M5AArtifactValidationError, match="local_files_only provenance differs"):
        validate_router_evaluation_evidence(
            root,
            expected_examples_by_split=examples_by_split,
        )
    result["local_files_only"] = True
    result["repeat_count"] = 3
    _write(root / "result.json", result)
    _refresh_router_checksums(root, owner)
    with pytest.raises(M5AArtifactValidationError, match="repeat_count differs"):
        validate_router_evaluation_evidence(
            root,
            expected_examples_by_split=examples_by_split,
        )
    result["repeat_count"] = 2
    _write(root / "result.json", result)
    _refresh_router_checksums(root, owner)

    record_path = root / "development" / "rule" / "records.jsonl"
    record_values = [json.loads(line) for line in record_path.read_text().splitlines()]
    record_index = next(
        index
        for index, value in enumerate(record_values)
        if value["expected_status"] == RouterStatus.ROUTE.value
    )
    record = record_values[record_index]
    original_decision = RouterDecision.from_dict(record["decision"])
    alternative = next(
        task_spec
        for task_spec in CANONICAL_TASK_SPECS
        if stable_task_id(task_spec) != record["expected_task_id"]
    )
    tampered_decision = RouterDecision.route(
        task_spec=alternative,
        confidence=original_decision.confidence,
        router_name=original_decision.router_name,
        router_version=original_decision.router_version,
        evidence=original_decision.evidence,
    )
    record["expected_task_id"] = stable_task_id(alternative)
    record["decision"] = tampered_decision.to_dict()
    record["repeat_decision_fingerprints"] = [
        tampered_decision.decision_fingerprint,
        tampered_decision.decision_fingerprint,
    ]
    record["status_correct"] = True
    record["full_task_correct"] = True
    record["exact_decision_correct"] = True
    record["deterministic"] = True
    record_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in record_values),
        encoding="utf-8",
    )
    _refresh_router_checksums(root, owner)
    with pytest.raises(M5AArtifactValidationError, match="authoritative corpus"):
        validate_router_evaluation_evidence(
            root,
            expected_examples_by_split=examples_by_split,
        )


@dataclass
class _FixtureEnvironment:
    reset_count: int = 0
    active_episode_spec: EpisodeSpec | None = None

    def reset(
        self, *, seed: int, options: Mapping[str, object]
    ) -> tuple[object, dict[str, object]]:
        assert seed >= 0 and isinstance(options["task_spec"], Mapping)
        self.reset_count += 1
        self.active_episode_spec = EpisodeSpec.create(
            scene_seed=seed,
            task_spec=TaskSpec.from_mapping(options["task_spec"]),
        )
        return {}, {}

    def step(self, action: np.ndarray) -> tuple[object, float, bool, bool, dict[str, object]]:
        assert action.shape == (8,)
        return {}, 0.0, False, False, {}

    def get_episode_specs(self) -> tuple[EpisodeSpec, ...]:
        assert self.active_episode_spec is not None
        return (self.active_episode_spec,)


def test_control_validator_rebuilds_288_records_summaries_and_completion(
    tmp_path: Path,
) -> None:
    command = _load_control_command()
    corpus = build_language_corpus()
    exclusions = tuple(
        SeedExclusionSource(
            source_id=source_id,
            source_fingerprint=f"sha256:{sha256_hex({'source': source_id})}",
            scene_seeds=(),
        )
        for source_id in REQUIRED_EXCLUSION_SOURCE_IDS
    )
    bundle = build_m5a_schedule_bundle(
        corpus=corpus,
        config=ControlScheduleConfig(exclusion_sources=exclusions),
    )
    inputs = command.prepare_development_inputs(
        schedule=bundle.control_development,
        corpus=corpus,
        stage=command.M5AStage.ONE_SCENE_CONTROL_SMOKE,
    )
    registry = build_fixture_controller_registry()
    environment = _FixtureEnvironment()
    loader = FixturePerTaskControllerLoader()
    dispatcher = ControllerDispatcher(
        registry=registry,
        loader=loader,
    )
    rejection_probe = command.run_rejection_noop_probe(
        registry=registry,
        loader=loader,
        environment=environment,
        rule_router=RuleRouterV0(),
    )

    def correct(router_name):
        def provide(episode, _example):
            return RouterDecision.route(
                task_spec=episode.task_spec,
                confidence=RouterConfidence.unavailable(),
                router_name=router_name,
                router_version="v0",
            )

        return provide

    result = command.run_development_control(
        inputs=inputs,
        registry=registry,
        dispatcher=dispatcher,
        environment=environment,
        providers={
            "oracle": command._oracle_provider,
            "classifier": correct("FactorizedTextClassifierV0"),
            "llm": correct("StructuredLocalLLMRouterV0"),
        },
        output_root=tmp_path / "control",
        run_identity={"fixture": "artifact-validator"},
        require_active_episode_spec=True,
        rejection_noop_probe=rejection_probe,
    )
    root = Path(result["evidence_root"])

    validated = validate_development_control_evidence(
        root,
        inputs=inputs,
        registry=registry,
    )
    assert validated.record_count == 18
    assert validated.record_set_fingerprint == result["record_set_fingerprint"]
    assert validated.summaries == result["summaries"]

    completion_path = root / "complete.json"
    original_completion = completion_path.read_bytes()

    def refresh_completion_from_atoms() -> None:
        record_fingerprints = []
        summaries = {}
        for router_label in CONTROL_ROUTER_ORDER:
            records = []
            for scheduled_episode in inputs.episodes:
                record = json.loads(
                    (
                        root
                        / "episodes"
                        / router_label
                        / f"{scheduled_episode.episode_index:03d}.json"
                    ).read_text(encoding="utf-8")
                )
                records.append(record)
                record_fingerprints.append(record["record_fingerprint"])
            summaries[router_label] = command._summary(records)
        refreshed = json.loads(original_completion)
        refreshed["summaries"] = summaries
        refreshed["record_set_fingerprint"] = f"sha256:{sha256_hex(record_fingerprints)}"
        for key in (
            "wrong_object_interaction_count",
            "invalid_action_count",
            "malformed_action_count",
            "nonfinite_action_count",
            "dispatch_after_rejection_count",
        ):
            refreshed[key] = sum(summary[key] for summary in summaries.values())
        refreshed_base = dict(refreshed)
        refreshed_base.pop("completion_fingerprint")
        refreshed["completion_fingerprint"] = f"sha256:{sha256_hex(refreshed_base)}"
        _write(completion_path, refreshed)

    completion = json.loads(original_completion)
    completion["summaries"]["oracle"]["episode_count"] = 71
    completion_base = dict(completion)
    completion_base.pop("completion_fingerprint")
    completion["completion_fingerprint"] = f"sha256:{sha256_hex(completion_base)}"
    _write(completion_path, completion)
    with pytest.raises(M5AArtifactValidationError, match="summaries differ"):
        validate_development_control_evidence(root, inputs=inputs, registry=registry)

    completion_path.write_bytes(original_completion)
    oracle_path = root / "episodes" / "oracle" / "000.json"
    original_oracle = oracle_path.read_bytes()
    oracle_record = json.loads(original_oracle)
    original_oracle_decision = RouterDecision.from_dict(oracle_record["result"]["decision"])
    nonoracle_decision = RouterDecision.route(
        task_spec=inputs.episodes[0].task_spec,
        confidence=original_oracle_decision.confidence,
        router_name="OracleTaskSpecRouterV0",
        router_version="v0",
        evidence=original_oracle_decision.evidence,
    )
    oracle_record["result"]["decision"] = nonoracle_decision.to_dict()
    oracle_record["result"]["dispatch"]["decision_fingerprint"] = (
        nonoracle_decision.decision_fingerprint
    )
    oracle_base = dict(oracle_record)
    oracle_base.pop("record_fingerprint")
    oracle_record["record_fingerprint"] = f"sha256:{sha256_hex(oracle_base)}"
    _write(oracle_path, oracle_record)
    refresh_completion_from_atoms()
    with pytest.raises(M5AArtifactValidationError, match="oracle control atom differs"):
        validate_development_control_evidence(root, inputs=inputs, registry=registry)

    oracle_path.write_bytes(original_oracle)
    completion_path.write_bytes(original_completion)
    episode_path = root / "episodes" / "classifier" / "000.json"
    original_episode = episode_path.read_bytes()
    episode = json.loads(original_episode)
    episode["active_episode_spec"]["canonical_instruction"] = "tampered instruction"
    episode_base = dict(episode)
    episode_base.pop("record_fingerprint")
    episode["record_fingerprint"] = f"sha256:{sha256_hex(episode_base)}"
    _write(episode_path, episode)
    refresh_completion_from_atoms()
    with pytest.raises(M5AArtifactValidationError, match="active EpisodeSpec differs"):
        validate_development_control_evidence(root, inputs=inputs, registry=registry)

    episode_path.write_bytes(original_episode)
    completion_path.write_bytes(original_completion)
    rejection_record = json.loads(original_episode)
    rejection = RouterDecision.reject(
        status=RouterStatus.REJECT_UNSUPPORTED,
        rejection_reason=RouterRejectionReason.UNSUPPORTED_ACTION,
        confidence=RouterConfidence.unavailable(),
        router_name="FactorizedTextClassifierV0",
        router_version="v0",
    )
    rejection_record["active_episode_spec"] = None
    rejection_record["result"]["decision"] = rejection.to_dict()
    rejection_record["result"]["dispatch"].update(
        {
            "decision_fingerprint": rejection.decision_fingerprint,
            "predicted_task_id": None,
            "controller_task_id": None,
            "controller_run_fingerprint": None,
            "controller_checkpoint_fingerprint": None,
            "dispatched": False,
            "controller_loaded": False,
            "policy_called": False,
            "environment_reset_called": False,
            "environment_step_count": 0,
            "safe_rejection": False,
            "controller_load_failed": False,
        }
    )
    rejection_record["result"]["control"] = None
    rejection_record["result"]["failure_attribution"] = "routing_false_rejection"
    rejection_record["result"]["end_to_end_success"] = False
    rejection_base = dict(rejection_record)
    rejection_base.pop("record_fingerprint")
    rejection_record["record_fingerprint"] = f"sha256:{sha256_hex(rejection_base)}"
    _write(episode_path, rejection_record)
    refresh_completion_from_atoms()
    with pytest.raises(M5AArtifactValidationError, match="safe zero-work rejection"):
        validate_development_control_evidence(root, inputs=inputs, registry=registry)

    episode_path.write_bytes(original_episode)
    completion_path.write_bytes(original_completion)
    episode = json.loads(original_episode)
    scheduled = inputs.episodes[0]
    original_decision = RouterDecision.from_dict(episode["result"]["decision"])
    alternative_entry = next(
        entry for entry in registry.entries if entry.task_id != original_decision.task_id
    )
    dispatch = episode["result"]["dispatch"]
    dispatch["predicted_task_id"] = alternative_entry.task_id
    dispatch["controller_task_id"] = alternative_entry.task_id
    dispatch["controller_run_fingerprint"] = alternative_entry.run_fingerprint
    dispatch["controller_checkpoint_fingerprint"] = alternative_entry.checkpoint_fingerprint
    control = episode["result"]["control"]
    attribution = attribute_end_to_end(
        FailureAttributionEvidence(
            expected_task_spec=scheduled.task_spec,
            decision=original_decision,
            controller_task_id=alternative_entry.task_id,
            controller_dispatched=True,
            control_success=control["success"],
        )
    )
    assert attribution is not None
    episode["result"]["failure_attribution"] = attribution.value
    episode["result"]["end_to_end_success"] = False
    episode_base = dict(episode)
    episode_base.pop("record_fingerprint")
    episode["record_fingerprint"] = f"sha256:{sha256_hex(episode_base)}"
    _write(episode_path, episode)

    refresh_completion_from_atoms()
    with pytest.raises(
        M5AArtifactValidationError,
        match="dispatch predicted task differs",
    ):
        validate_development_control_evidence(root, inputs=inputs, registry=registry)
