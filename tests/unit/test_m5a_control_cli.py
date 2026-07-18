from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import EpisodeSpec, TaskSpec
from langmani.language.controller_registry import build_fixture_controller_registry
from langmani.language.corpus import build_language_corpus
from langmani.language.dispatcher import (
    ControllerDispatcher,
    FixturePerTaskControllerLoader,
)
from langmani.language.router_types import (
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterV0
from langmani.language.schedules import (
    M5A_CONTROL_FINAL_SCHEDULE_ID,
    M5A_TARGET_DEVELOPMENT_SCHEDULE_LOCK_SCHEMA,
    REQUIRED_EXCLUSION_SOURCE_IDS,
    ControlScheduleConfig,
    SeedExclusionSource,
    build_m5a_schedule_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_command() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_language_control.py"
    spec = importlib.util.spec_from_file_location("langmani_test_run_language_control", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _SpyEnvironment:
    reset_count: int = 0
    step_count: int = 0
    reset_tasks: list[dict[str, str]] = field(default_factory=list)
    last_seed: int | None = None

    def reset(self, *, seed: int, options: dict[str, object]) -> tuple[object, dict[str, object]]:
        assert seed >= 0
        self.last_seed = seed
        self.reset_count += 1
        self.reset_tasks.append(dict(options["task_spec"]))
        return {}, {}

    def step(self, action: np.ndarray) -> tuple[object, float, bool, bool, dict[str, object]]:
        assert action.shape == (8,)
        self.step_count += 1
        return {}, 0.0, False, False, {}

    def get_episode_specs(self) -> tuple[EpisodeSpec, ...]:
        assert self.last_seed is not None and self.reset_tasks
        return (
            EpisodeSpec.create(
                scene_seed=self.last_seed,
                task_spec=TaskSpec.from_mapping(self.reset_tasks[-1]),
            ),
        )


def _control_schedule_config() -> ControlScheduleConfig:
    exclusions = tuple(
        SeedExclusionSource(
            source_id=source_id,
            source_fingerprint=f"sha256:{sha256_hex({'source': source_id})}",
            scene_seeds=(),
        )
        for source_id in REQUIRED_EXCLUSION_SOURCE_IDS
    )
    return ControlScheduleConfig(exclusion_sources=exclusions)


def _bundle():
    corpus = build_language_corpus()
    return corpus, build_m5a_schedule_bundle(
        corpus=corpus,
        config=_control_schedule_config(),
    )


def _authoritative_schedule_wrapper() -> tuple[object, object, dict[str, object]]:
    corpus = build_language_corpus()
    config = _control_schedule_config()
    bundle = build_m5a_schedule_bundle(corpus=corpus, config=config)
    language_final = bundle.language_final
    control_final = bundle.control_final
    payload: dict[str, object] = {
        "schema_version": M5A_TARGET_DEVELOPMENT_SCHEDULE_LOCK_SCHEMA,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "m43b_evaluation_evidence_fingerprint": f"sha256:{'f' * 64}",
        "control_schedule_config": config.to_dict(),
        "language_development": bundle.language_development.to_dict(),
        "language_final": {
            "schedule_id": language_final.schedule_id,
            "schedule_fingerprint": language_final.schedule_fingerprint,
            "corpus_fingerprint": language_final.corpus_fingerprint,
            "split_fingerprint": language_final.split_fingerprint,
            "example_count": len(language_final.ordered_example_ids),
            "sealed": True,
            "texts_materialized": False,
        },
        "control_development": bundle.control_development.to_dict(),
        "control_final": {
            "schedule_id": control_final.schedule_id,
            "schedule_fingerprint": control_final.schedule_fingerprint,
            "ordered_scene_seeds": list(control_final.ordered_scene_seeds),
            "scene_count": len(control_final.ordered_scene_seeds),
            "episode_count": control_final.episode_count,
            "exclusion_digest": control_final.exclusion_digest,
            "language_schedule_fingerprint": control_final.language_schedule_fingerprint,
            "sealed": True,
            "language_example_ids_materialized": False,
            "episodes_materialized": False,
        },
        "all_four_schedules_locked_before_training": True,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "smolvla_go": False,
        "final_texts_materialized": False,
        "final_episodes_materialized": False,
    }
    return corpus, bundle, payload


def _router_evaluation_fixture(
    command: ModuleType,
    root: Path,
    *,
    corpus,
    classifier_identity: dict[str, object],
    llm_config,
    mutate_prompt: bool = False,
) -> Path:
    root.mkdir()
    examples = command.select_structured_routing_prompt_examples(
        corpus.examples_for_split(command.LanguageSplit.TRAIN)
    )
    template, prompt_fingerprint, prompt_ids = command.build_structured_routing_prompt(
        examples=examples,
        prompt_version=llm_config.prompt_version,
    )
    if mutate_prompt:
        prompt_ids = tuple(reversed(prompt_ids))
    evaluation_fingerprint = f"sha256:{'e' * 64}"
    owner = {"evaluation_fingerprint": evaluation_fingerprint}
    result = {
        "passed": True,
        "mode": "target_development",
        "target_development_executed": True,
        "language_development_completed": True,
        "evaluation_fingerprint": evaluation_fingerprint,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "router_identities": {
            "rule": {
                "router_fingerprint": command.RuleRouterV0().config.fingerprint,
            },
            "classifier": {
                "artifact_fingerprint": classifier_identity["artifact_fingerprint"],
                "run_evidence_fingerprint": classifier_identity["run_evidence_fingerprint"],
                "router_config": {
                    **classifier_identity["router_config"],
                    "device": "cuda",
                },
            },
            "llm": {
                "config": llm_config.to_dict(),
                "declared_license": "Apache-2.0",
                "official_model_card_license_reviewed": True,
                "prompt_template": template,
                "prompt_fingerprint": prompt_fingerprint,
                "prompt_example_ids": list(prompt_ids),
            },
        },
        "candidate_promotions": {
            "classifier": {"promoted": True},
            "llm": {"promoted": True},
        },
        "promoted_learned_routers": ["classifier", "llm"],
        "primary_learned_router": "classifier",
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "smolvla_go": False,
    }
    (root / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    artifacts = command._router_evaluation_artifacts(root)
    artifact_fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': artifacts})}"
    (root / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifact_fingerprint": artifact_fingerprint,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    (root / "complete.json").write_text(
        json.dumps(
            {
                "passed": True,
                "evaluation_fingerprint": evaluation_fingerprint,
                "artifact_fingerprint": artifact_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_control_reuses_exact_immutable_language_evaluation_prompt(tmp_path: Path) -> None:
    command = _load_command()
    corpus = build_language_corpus()
    classifier_identity = {
        "artifact_fingerprint": f"sha256:{'a' * 64}",
        "run_evidence_fingerprint": f"sha256:{'b' * 64}",
        "router_config": {
            "maximum_sequence_length": 64,
            "routing_threshold": 0.8,
        },
    }
    llm_config = command.StructuredLLMRouterConfig(
        model_id="fixture/local-instruct",
        model_revision="c" * 40,
        tokenizer_revision="d" * 40,
    )
    root = _router_evaluation_fixture(
        command,
        tmp_path / "valid",
        corpus=corpus,
        classifier_identity=classifier_identity,
        llm_config=llm_config,
    )

    examples, identity = command._load_frozen_router_evaluation(
        root,
        corpus=corpus,
        classifier_identity=classifier_identity,
        rule=command.RuleRouterV0(),
        llm_config=llm_config,
        llm_license="Apache-2.0",
        llm_license_reviewed=True,
    )

    assert [example.example_id for example in examples] == identity["router_identities"]["llm"][
        "prompt_example_ids"
    ]

    wrong = _router_evaluation_fixture(
        command,
        tmp_path / "wrong",
        corpus=corpus,
        classifier_identity=classifier_identity,
        llm_config=llm_config,
        mutate_prompt=True,
    )
    with pytest.raises(command.LanguageControlCommandError, match="not the canonical train prompt"):
        command._load_frozen_router_evaluation(
            wrong,
            corpus=corpus,
            classifier_identity=classifier_identity,
            rule=command.RuleRouterV0(),
            llm_config=llm_config,
            llm_license="Apache-2.0",
            llm_license_reviewed=True,
        )


def test_control_cli_rejects_final_lock_before_materialization(tmp_path: Path) -> None:
    command = _load_command()
    _corpus, bundle = _bundle()
    path = tmp_path / "final.json"
    path.write_text(json.dumps(bundle.control_final.to_dict()), encoding="utf-8")

    with pytest.raises(command.LanguageControlCommandError, match="only m5a_control_dev_v0"):
        command.load_development_schedule(path)

    assert bundle.control_final.schedule_id == M5A_CONTROL_FINAL_SCHEDULE_ID


def test_authoritative_schedule_loader_rebuilds_all_four_locks(tmp_path: Path) -> None:
    command = _load_command()
    corpus, bundle, payload = _authoritative_schedule_wrapper()
    path = tmp_path / "target-development-schedules.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = command.load_authoritative_development_schedule(path, corpus=corpus)

    assert loaded == bundle.control_development


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_exclusion", "every required source"),
        ("language_final_ids", "language_final differs"),
        ("control_final_ids", "control_final differs"),
        ("final_materialized", "final/test/fresh access locks"),
        ("fingerprint_drift", "control_development differs"),
        ("config_drift", "control_development differs"),
    ),
)
def test_authoritative_schedule_loader_rejects_non_authoritative_bundle(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    command = _load_command()
    corpus, _bundle_value, source = _authoritative_schedule_wrapper()
    payload = json.loads(json.dumps(source))
    if mutation == "missing_exclusion":
        payload["control_schedule_config"]["exclusion_sources"].pop()
    elif mutation == "language_final_ids":
        payload["language_final"]["ordered_example_ids"] = ["forbidden-final-id"]
    elif mutation == "control_final_ids":
        payload["control_final"]["associated_language_example_ids"] = ["forbidden-final-id"]
    elif mutation == "final_materialized":
        payload["final_episodes_materialized"] = True
    elif mutation == "fingerprint_drift":
        payload["control_development"]["schedule_fingerprint"] = f"sha256:{'0' * 64}"
    elif mutation == "config_drift":
        payload["control_schedule_config"]["candidate_seed_start"] += 1
        config_payload = payload["control_schedule_config"]
        config = command._parse_authoritative_control_config(
            {
                **config_payload,
                "exclusion_digest": _control_schedule_config().exclusion_digest,
            }
        )
        payload["control_schedule_config"] = config.to_dict()
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)
    path = tmp_path / f"{mutation}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(command.LanguageControlCommandError, match=message):
        command.load_authoritative_development_schedule(path, corpus=corpus)


def test_execute_requires_authoritative_wrapper_but_direct_loader_remains_available(
    tmp_path: Path,
) -> None:
    command = _load_command()
    corpus, bundle, payload = _authoritative_schedule_wrapper()
    wrapper = tmp_path / "target-development-schedules.json"
    wrapper.write_text(json.dumps(payload), encoding="utf-8")
    direct = tmp_path / "direct-development-lock.json"
    direct.write_text(json.dumps(bundle.control_development.to_dict()), encoding="utf-8")
    assert command.load_development_schedule(direct) == bundle.control_development

    common = [
        "--classifier-artifact-root",
        str(tmp_path / "classifier"),
        "--router-evaluation-evidence-root",
        str(tmp_path / "router-evaluation"),
        "--llm-model-id",
        "fixture/local",
        "--llm-model-revision",
        "a" * 40,
        "--llm-tokenizer-revision",
        "b" * 40,
        "--llm-license",
        "Apache-2.0",
        "--output-root",
        str(tmp_path / "output"),
        "--report",
        str(tmp_path / "report.json"),
        "--stage",
        "one_scene_control_smoke",
        "--dry-run",
    ]
    result = command.execute(command.parse_args(["--control-schedule", str(wrapper), *common]))
    assert result["passed"] is True
    assert result["schedule_fingerprint"] == bundle.one_scene_control_smoke.schedule_fingerprint

    with pytest.raises(
        command.LanguageControlCommandError,
        match="authoritative schema",
    ):
        command.execute(command.parse_args(["--control-schedule", str(direct), *common]))
    assert corpus.manifest.corpus_fingerprint == result["corpus_fingerprint"]


def test_one_scene_promoted_control_is_ordered_resumable_and_rejection_is_zero_step(
    tmp_path: Path,
) -> None:
    command = _load_command()
    corpus, bundle = _bundle()
    inputs = command.prepare_development_inputs(
        schedule=bundle.control_development,
        corpus=corpus,
        stage=command.M5AStage.ONE_SCENE_CONTROL_SMOKE,
    )
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    environment = _SpyEnvironment()
    dispatcher = ControllerDispatcher(registry=registry, loader=loader)
    provider_calls: list[tuple[str, int]] = []

    def oracle(episode, _example):
        provider_calls.append(("oracle", episode.episode_index))
        return command._oracle_provider(episode, _example)

    def classifier(episode, _example):
        provider_calls.append(("classifier", episode.episode_index))
        if episode.episode_index == 0:
            return RouterDecision.reject(
                status=RouterStatus.REJECT_AMBIGUOUS,
                rejection_reason=RouterRejectionReason.LOW_CONFIDENCE,
                confidence=RouterConfidence.probability(
                    0.2,
                    definition="fixture calibrated confidence",
                    calibrated=True,
                ),
                router_name="FactorizedTextClassifierV0",
                router_version="fixture-v0",
            )
        return RouterDecision.route(
            task_spec=episode.task_spec,
            confidence=RouterConfidence.probability(
                0.9,
                definition="fixture calibrated confidence",
                calibrated=True,
            ),
            router_name="FactorizedTextClassifierV0",
            router_version="fixture-v0",
        )

    def llm(episode, _example):
        provider_calls.append(("llm", episode.episode_index))
        return RouterDecision.route(
            task_spec=episode.task_spec,
            confidence=RouterConfidence.unavailable(),
            router_name="StructuredLocalLLMRouterV0",
            router_version="fixture-v0",
        )

    providers = {
        "oracle": oracle,
        "classifier": classifier,
        "llm": llm,
    }
    result = command.run_development_control(
        inputs=inputs,
        registry=registry,
        dispatcher=dispatcher,
        environment=environment,
        providers=providers,
        output_root=tmp_path / "control",
        run_identity={"fixture": True},
        require_active_episode_spec=True,
    )

    assert result["passed"] is True
    assert result["completed_episode_atoms"] == 18
    assert provider_calls == [
        (router, episode_index)
        for router in ("oracle", "classifier", "llm")
        for episode_index in range(6)
    ]
    assert environment.reset_count == environment.step_count == 17
    assert result["summaries"]["classifier"]["safe_rejection_count"] == 1
    assert result["summaries"]["classifier"]["dispatched_count"] == 5
    assert result["summaries"]["classifier"]["wrong_object_interaction_available"] is True
    assert result["summaries"]["classifier"]["wrong_object_interaction_count"] == 0
    assert result["summaries"]["classifier"]["invalid_action_count"] == 0
    assert result["summaries"]["classifier"]["malformed_action_count"] == 0
    assert result["summaries"]["classifier"]["nonfinite_action_count"] == 0
    assert result["summaries"]["classifier"]["dispatch_after_rejection_count"] == 0
    assert result["summaries"]["classifier"]["m2_expert_call_count"] == 0
    assert result["wrong_object_interaction_available"] is True
    assert result["wrong_object_interaction_count"] == 0
    assert result["dispatch_after_rejection_count"] == 0
    assert result["zero_dispatch_after_rejection_validated"] is True
    assert result["m2_expert_call_count"] == 0
    assert result["m2_expert_free_validated"] is True
    assert result["summaries"]["oracle"]["episode_count"] == 6
    assert result["summaries"]["llm"]["episode_count"] == 6
    assert result["promoted_to_next_stage"] == ["llm"]

    provider_calls.clear()
    before_steps = environment.step_count
    reused = command.run_development_control(
        inputs=inputs,
        registry=registry,
        dispatcher=dispatcher,
        environment=environment,
        providers=providers,
        output_root=tmp_path / "control",
        run_identity={"fixture": True},
        require_active_episode_spec=True,
    )

    assert reused["reused"] is True
    assert provider_calls == []
    assert environment.step_count == before_steps


def test_target_rejection_probe_uses_real_environment_boundary_without_work() -> None:
    command = _load_command()
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    environment = _SpyEnvironment()

    probe = command.run_rejection_noop_probe(
        registry=registry,
        loader=loader,
        environment=environment,
        rule_router=RuleRouterV0(),
    )

    assert probe["passed"] is True
    assert probe["physical_m1_environment"] is True
    assert probe["controller_lookup_count"] == 0
    assert probe["environment_reset_count"] == 0
    assert probe["environment_step_count"] == 0
    assert loader.load_count == 0
    assert environment.reset_count == environment.step_count == 0
    command._validate_rejection_noop_probe(probe)


def test_completed_control_evidence_is_checksum_validated(tmp_path: Path) -> None:
    command = _load_command()
    corpus, bundle = _bundle()
    inputs = command.prepare_development_inputs(
        schedule=bundle.control_development,
        corpus=corpus,
        stage=command.M5AStage.ONE_SCENE_CONTROL_SMOKE,
    )
    registry = build_fixture_controller_registry()
    environment = _SpyEnvironment()
    dispatcher = ControllerDispatcher(
        registry=registry,
        loader=FixturePerTaskControllerLoader(),
    )

    def correct(episode, _example):
        return RouterDecision.route(
            task_spec=episode.task_spec,
            confidence=RouterConfidence.unavailable(),
            router_name="fixture",
            router_version="v0",
        )

    result = command.run_development_control(
        inputs=inputs,
        registry=registry,
        dispatcher=dispatcher,
        environment=environment,
        providers={name: correct for name in ("oracle", "classifier", "llm")},
        output_root=tmp_path / "control",
        run_identity={"fixture": "checksum"},
    )
    completion = Path(result["evidence_root"]) / "complete.json"
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["completed_episode_atoms"] = 1
    completion.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(command.LanguageControlCommandError, match="fingerprint differs"):
        command.run_development_control(
            inputs=inputs,
            registry=registry,
            dispatcher=dispatcher,
            environment=environment,
            providers={name: correct for name in ("oracle", "classifier", "llm")},
            output_root=tmp_path / "control",
            run_identity={"fixture": "checksum"},
        )


def test_control_summary_counts_runtime_safety_signals() -> None:
    command = _load_command()
    summary = command._summary(
        [
            {
                "router_inference_latency_ms": 2.0,
                "result": {
                    "decision": {
                        "status": "reject_ambiguous",
                        "task_id": None,
                        "target_object_id": None,
                        "target_bin_id": None,
                    },
                    "dispatch": {
                        "oracle_task_id": "oracle-task",
                        "dispatched": True,
                        "controller_loaded": True,
                        "policy_called": True,
                        "environment_reset_called": True,
                        "environment_step_count": 1,
                        "safe_rejection": False,
                        "controller_load_failed": False,
                    },
                    "oracle_task_spec": {
                        "target_object_id": "red_cube",
                        "target_bin_id": "left_bin",
                        "instruction_template_id": "canonical_v0",
                    },
                    "end_to_end_success": False,
                    "failure_attribution": "routing_false_rejection",
                    "control": {
                        "success": False,
                        "invalid_action": True,
                        "wrong_object_interaction": True,
                        "timeout": False,
                        "infrastructure_failed": False,
                        "controller_inference_failed": False,
                        "final_evaluation": {
                            "wrong_object_in_target_bin": True,
                            "target_in_wrong_bin": False,
                            "target_off_table": False,
                        },
                        "action_evidence": {
                            "raw_violation_count": 1,
                            "arm_projected_component_count": 2,
                            "gripper_projected_component_count": 1,
                            "malformed_action_count": 3,
                            "nonfinite_action_count": 4,
                        },
                        "inference_latency_ms": [1.0],
                        "environment_latency_ms": [5.0],
                    },
                },
            }
        ]
    )

    assert summary["wrong_object_interaction_available"] is True
    assert summary["wrong_object_interaction_count"] == 1
    assert summary["invalid_action_count"] == 1
    assert summary["malformed_action_count"] == 3
    assert summary["nonfinite_action_count"] == 4
    assert summary["dispatch_after_rejection_count"] == 1
    assert summary["m2_expert_call_count"] == 0


def test_control_command_has_no_m2_expert_dependency() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_language_control.py").read_text(encoding="utf-8")

    assert "langmani.experts" not in source
    assert "PickPlaceExpert" not in source


def test_controller_registry_gate_is_metadata_only_and_keeps_six_provenance_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _load_command()
    registry = build_fixture_controller_registry()
    locators = object()
    calls: list[dict[str, object]] = []

    def load_metadata(**kwargs: object) -> tuple[object, object]:
        calls.append(dict(kwargs))
        return registry, locators

    monkeypatch.setattr(command, "load_controller_registry_metadata", load_metadata)

    loaded_registry, loaded_locators = command._load_frozen_controller_registry(
        checkpoint_root=tmp_path / "checkpoints",
        dataset_root=tmp_path / "m3b",
        runtime_selection_path=tmp_path / "runtime-selection.json",
    )

    assert loaded_registry is registry
    assert loaded_locators is locators
    assert len(loaded_registry.entries) == 6
    assert len({entry.checkpoint_fingerprint for entry in loaded_registry.entries}) == 6
    assert calls == [
        {
            "checkpoint_root": tmp_path / "checkpoints",
            "dataset_root": tmp_path / "m3b",
            "runtime_selection_path": tmp_path / "runtime-selection.json",
        }
    ]
