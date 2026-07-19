"""Run one promoted M5A language-to-frozen-controller development stage.

This command deliberately has no final-schedule mode.  It always evaluates the
oracle and only the learned candidates promoted by the preceding immutable
stage: 1 scene x 6 tasks for smoke, 3 disjoint scenes x 6 tasks for screening,
or 6 further scenes x 6 tasks for the single selected router.  RuleRouterV0
remains an offline baseline.  Episode records are immutable and individually
resumable; a completed stage is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from langmani.datasets.identity import sha256_hex  # noqa: E402
from langmani.datasets.schedule import CANONICAL_TASK_SPECS  # noqa: E402
from langmani.environments.specs import TaskSpec, stable_task_id  # noqa: E402
from langmani.language.controller_registry import (  # noqa: E402
    ControllerRegistry,
    ControllerRegistryLocators,
    load_controller_registry_metadata,
)
from langmani.language.corpus import (  # noqa: E402
    GeneratedLanguageCorpus,
    build_language_corpus,
)
from langmani.language.dispatcher import (  # noqa: E402
    ControllerDispatcher,
    StrictPerTaskControllerLoader,
)
from langmani.language.failure_attribution import FailureAttribution  # noqa: E402
from langmani.language.full_control_development import (  # noqa: E402
    FULL_REJECTION_CASES,
    analyze_full_control_records,
    build_final_authorization,
    build_full_rejection_probe_set,
)
from langmani.language.full_control_evidence import (  # noqa: E402
    write_full_control_evidence,
)
from langmani.language.llm_router import (  # noqa: E402
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
    TransformersLocalTextGenerator,
    build_structured_routing_prompt,
    select_structured_routing_prompt_examples,
)
from langmani.language.neuro_symbolic_control_verifier import (  # noqa: E402
    verify_neuro_symbolic_control_evidence,
)
from langmani.language.neuro_symbolic_dispatch import (  # noqa: E402
    NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL,
    SelectedNeuroSymbolicDispatchSource,
    bind_selected_router_to_controller_registry,
    load_selected_neuro_symbolic_router,
    validate_selected_neuro_symbolic_dispatch_source,
)
from langmani.language.router_types import (  # noqa: E402
    LanguageExample,
    LanguageSplit,
    RouterConfidence,
    RouterDecision,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterV0  # noqa: E402
from langmani.language.schedules import (  # noqa: E402
    M5A_CONTROL_DEV_SCHEDULE_ID,
    M5A_CONTROL_FINAL_SCHEDULE_ID,
    M5A_DEVELOPMENT_SCENE_COUNT,
    M5A_FINAL_SCENE_COUNT,
    M5A_LEGACY_DEVELOPMENT_SCENE_COUNT,
    M5A_LEGACY_FINAL_SCENE_COUNT,
    M5A_TARGET_DEVELOPMENT_SCHEDULE_LOCK_SCHEMA,
    ControlScheduleConfig,
    ControlScheduleLock,
    LanguageScheduleLock,
    M5AScheduleBundle,
    ScheduledControlEpisode,
    SeedExclusionSource,
    StagedControlSchedule,
    build_language_schedule_locks,
    build_staged_control_schedule,
)
from langmani.language.stage_protocol import M5A_PHYSICAL_STAGE_BUDGETS, M5AStage  # noqa: E402
from langmani.language.text_calibration import TemperatureCalibrationV0  # noqa: E402
from langmani.language.text_classifier import (  # noqa: E402
    FactorizedTextRouterConfig,
    FactorizedTextRouterV0,
    load_factorized_text_classifier,
)
from langmani.language.text_training import read_text_classifier_run_evidence  # noqa: E402
from langmani.language.three_scene_control import (  # noqa: E402
    THREE_SCENE_REJECTION_CASES,
    analyze_three_scene_records,
    build_rejection_probe_set,
    initial_scene_state_fingerprint,
    normalize_initial_scene_state,
)
from langmani.language.three_scene_control_evidence import (  # noqa: E402
    write_three_scene_evidence,
)
from langmani.language.three_scene_control_verifier import (  # noqa: E402
    verify_three_scene_control_evidence,
)
from langmani.policies.act_runtime import inspect_git_state  # noqa: E402

COMMAND_SCHEMA = "langmani-m5a-language-control-command-v0"
RUN_SCHEMA = "langmani-m5a-language-control-run-v0"
EPISODE_SCHEMA = "langmani-m5a-language-control-episode-v0"
COMPLETION_SCHEMA = "langmani-m5a-language-control-complete-v0"
REJECTION_NOOP_PROBE_SCHEMA = "langmani-m5a-rejection-noop-probe-v0"
ROUTER_ORDER = ("oracle", "rule", "classifier", "llm", "neuro_symbolic")
LEARNED_ROUTER_ORDER = ("classifier", "llm", NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL)
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "models" / "act"
DEFAULT_RUNTIME_SELECTION = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "runtime_ablation" / "runtime_selection.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "control"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "run-language-control.json"
PROTECTED_ROOTS = tuple(
    PROJECT_ROOT / name for name in ("src", "scripts", "environment", "tests", "docs", ".git")
)
PROTECTED_GENERATED_ROOTS = (
    PROJECT_ROOT / "outputs" / "datasets",
    PROJECT_ROOT / "outputs" / "models",
    *(
        PROJECT_ROOT / "outputs" / "diagnostics" / name
        for name in ("m0", "m1", "m2", "m3a", "m3b", "m4", "m42", "m43")
    ),
)


class LanguageControlCommandError(RuntimeError):
    """Raised when a development run is unsafe, incomplete, or inconsistent."""


class RouterLike(Protocol):
    """Minimal router boundary shared by the three real router implementations."""

    def route(self, command: str) -> RouterDecision: ...


DecisionProvider = Callable[[ScheduledControlEpisode, LanguageExample], RouterDecision]


class _CountingControllerLoader:
    """Count strict-loader entry without changing its behavior."""

    def __init__(self, inner: StrictPerTaskControllerLoader) -> None:
        self.inner = inner
        self.lookup_count = 0

    def load(self, entry, *, environment):  # type: ignore[no-untyped-def]
        self.lookup_count += 1
        return self.inner.load(entry, environment=environment)


class _CountingEnvironmentProxy:
    """Expose the real M1 environment while counting forbidden rejection work."""

    def __init__(self, environment: object) -> None:
        self.environment = environment
        self.reset_count = 0
        self.step_count = 0

    def reset(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.reset_count += 1
        return cast(Any, self.environment).reset(*args, **kwargs)

    def step(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.step_count += 1
        return cast(Any, self.environment).step(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self.environment, name)


class _PairedInitialStateEnvironment:
    """Capture the privileged reset snapshot before each paired rollout begins."""

    def __init__(self, environment: object) -> None:
        self.environment = environment
        self._capture: dict[str, object] | None = None
        self._reset_count = 0

    def begin_episode(self) -> None:
        self._capture = None
        self._reset_count = 0

    def reset(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = cast(Any, self.environment).reset(*args, **kwargs)
        base = getattr(self.environment, "unwrapped", self.environment)
        accessor = getattr(base, "get_expert_initial_scene_state", None)
        if not callable(accessor):
            raise LanguageControlCommandError(
                "three-scene pairing requires get_expert_initial_scene_state()"
            )
        state = normalize_initial_scene_state(accessor())
        self._capture = {
            "initial_physical_state": state,
            "initial_physical_state_fingerprint": initial_scene_state_fingerprint(state),
        }
        self._reset_count += 1
        return result

    def consume(self, *, dispatched: bool) -> dict[str, object] | None:
        if not dispatched:
            if self._reset_count != 0 or self._capture is not None:
                raise LanguageControlCommandError(
                    "a rejected decision reset the paired environment"
                )
            return None
        if self._reset_count != 1 or self._capture is None:
            raise LanguageControlCommandError(
                "each dispatched paired episode must perform exactly one captured reset"
            )
        return dict(self._capture)

    def __getattr__(self, name: str) -> object:
        return getattr(self.environment, name)


@dataclass(frozen=True, slots=True)
class DevelopmentControlInputs:
    """Validated development schedule and command lookup."""

    schedule: StagedControlSchedule
    episodes: tuple[ScheduledControlEpisode, ...]
    examples_by_id: Mapping[str, LanguageExample]
    corpus_fingerprint: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-schedule",
        type=Path,
        required=True,
        help=(
            "Authoritative target-development four-schedule lock wrapper produced before "
            "router training."
        ),
    )
    parser.add_argument("--classifier-artifact-root", type=Path)
    parser.add_argument(
        "--router-evaluation-evidence-root",
        type=Path,
        help="Immutable evidence root produced by evaluate_language_routers.py.",
    )
    parser.add_argument("--llm-model-id")
    parser.add_argument("--llm-model-revision")
    parser.add_argument("--llm-tokenizer-revision")
    parser.add_argument("--llm-license")
    parser.add_argument("--llm-license-reviewed", action="store_true")
    parser.add_argument(
        "--llm-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--llm-maximum-new-tokens", type=int, default=128)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--neuro-symbolic-evidence-root",
        type=Path,
        help=(
            "Selected immutable M5A.4.1 evidence. This mode is local-cache-only and is "
            "authorized only through the explicit staged physical gates."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--m4-diagnostics-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--runtime-selection", type=Path, default=DEFAULT_RUNTIME_SELECTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--stage",
        choices=(
            M5AStage.ONE_SCENE_CONTROL_SMOKE.value,
            M5AStage.THREE_SCENE_CONTROL_SCREEN.value,
            M5AStage.FULL_CONTROL_DEVELOPMENT.value,
        ),
        required=True,
    )
    parser.add_argument(
        "--prior-stage-report",
        type=Path,
        help="Required for the three-scene screen and full-development stages.",
    )
    parser.add_argument(
        "--prior-stage-verification",
        type=Path,
        help=(
            "Independent verification of the immediately preceding one-scene or three-scene stage."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _router_source_mode(args: argparse.Namespace) -> str:
    """Require exactly one immutable router lineage before any output or runtime work."""

    legacy_values = (
        args.classifier_artifact_root,
        args.router_evaluation_evidence_root,
        args.llm_model_id,
        args.llm_model_revision,
        args.llm_tokenizer_revision,
        args.llm_license,
    )
    if args.neuro_symbolic_evidence_root is not None:
        if any(value is not None for value in legacy_values):
            raise LanguageControlCommandError(
                "neuro-symbolic dispatch cannot mix legacy classifier/LLM evidence"
            )
        if args.allow_model_download:
            raise LanguageControlCommandError(
                "neuro-symbolic dispatch requires the immutable local model cache"
            )
        one_scene = args.stage == M5AStage.ONE_SCENE_CONTROL_SMOKE.value
        three_scene = args.stage == M5AStage.THREE_SCENE_CONTROL_SCREEN.value
        full_development = args.stage == M5AStage.FULL_CONTROL_DEVELOPMENT.value
        if one_scene and (
            args.prior_stage_report is not None or args.prior_stage_verification is not None
        ):
            raise LanguageControlCommandError(
                "one-scene control does not accept prior-stage evidence"
            )
        if three_scene and (
            args.prior_stage_report is None or args.prior_stage_verification is None
        ):
            raise LanguageControlCommandError(
                "three-scene control requires the one-scene report and independent verification"
            )
        if full_development and (
            args.prior_stage_report is None or args.prior_stage_verification is None
        ):
            raise LanguageControlCommandError(
                "full control development requires the three-scene report and verification"
            )
        if not one_scene and not three_scene and not full_development:
            raise LanguageControlCommandError(
                "the selected M5A.4.1 router is not authorized for this physical stage"
            )
        return NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL
    if any(value is None for value in legacy_values):
        raise LanguageControlCommandError(
            "legacy dispatch requires classifier, router-evaluation, and complete LLM identity"
        )
    return "legacy"


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise LanguageControlCommandError(f"{label} traverses a symlink or junction")
    return lexical.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _safe_output_paths(
    output_root: Path,
    report: Path,
    *,
    immutable_inputs: Mapping[str, Path] | None = None,
) -> tuple[Path, Path]:
    output = _resolved_unlinked(output_root, label="M5A control output")
    report_path = _resolved_unlinked(report, label="M5A control report")
    protected = tuple(
        _resolved_unlinked(path, label="protected repository or historical artifact")
        for path in (*PROTECTED_ROOTS, *PROTECTED_GENERATED_ROOTS)
    )
    resolved_inputs = tuple(
        (
            label,
            _resolved_unlinked(path, label=f"immutable {label} input"),
        )
        for label, path in (immutable_inputs or {}).items()
    )
    for index, (left_label, left_path) in enumerate(resolved_inputs):
        for right_label, right_path in resolved_inputs[index + 1 :]:
            if _overlaps(left_path, right_path):
                raise LanguageControlCommandError(
                    f"immutable {left_label} and {right_label} inputs cannot overlap"
                )
    input_paths = tuple(path for _label, path in resolved_inputs)
    if any(_overlaps(output, path) for path in (*protected, *input_paths)):
        raise LanguageControlCommandError(
            "control output cannot overlap source, Git, historical artifacts, or immutable inputs"
        )
    if any(_overlaps(report_path, path) for path in (*protected, *input_paths, output)):
        raise LanguageControlCommandError(
            "control report cannot overlap source, Git, historical artifacts, immutable inputs, "
            "or control output"
        )
    if output.exists() and (not output.is_dir() or output.is_symlink() or output.is_junction()):
        raise LanguageControlCommandError("control output root must be one real directory")
    if report_path.exists() and (
        not report_path.is_file() or report_path.is_symlink() or report_path.is_junction()
    ):
        raise LanguageControlCommandError("control report must be one real file when present")
    return output, report_path


def _safe_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    immutable_inputs = {
        "control schedule": args.control_schedule,
        "M3B dataset": args.dataset_root,
        "ACT checkpoint": args.checkpoint_root,
        "runtime selection": args.runtime_selection,
    }
    if args.classifier_artifact_root is not None:
        immutable_inputs["classifier artifact"] = args.classifier_artifact_root
    if args.router_evaluation_evidence_root is not None:
        immutable_inputs["router evaluation evidence"] = args.router_evaluation_evidence_root
    if args.neuro_symbolic_evidence_root is not None:
        immutable_inputs["neuro-symbolic evidence"] = args.neuro_symbolic_evidence_root
    if args.prior_stage_report is not None:
        immutable_inputs["prior-stage report"] = args.prior_stage_report
    if args.prior_stage_verification is not None:
        immutable_inputs["prior-stage verification"] = args.prior_stage_verification
    if args.m4_diagnostics_root is not None:
        immutable_inputs["legacy M4 diagnostics"] = args.m4_diagnostics_root
    return _safe_output_paths(
        args.output_root,
        args.report,
        immutable_inputs=immutable_inputs,
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or path.is_junction() or not path.is_file():
        raise LanguageControlCommandError(f"unsafe or missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LanguageControlCommandError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise LanguageControlCommandError(f"{label} must contain one JSON object")
    return cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _router_evaluation_artifacts(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or candidate.is_junction():
            raise LanguageControlCommandError(
                f"router-evaluation evidence contains a linked entry: {candidate}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise LanguageControlCommandError(
                f"router-evaluation evidence contains a special entry: {candidate}"
            )
        relative = PurePosixPath(candidate.relative_to(root)).as_posix()
        if relative in {"artifact_manifest.json", "complete.json"}:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    return sorted(records, key=lambda value: cast(str, value["path"]))


def _write_immutable_json(path: Path, value: Mapping[str, object]) -> bool:
    """Create one fsync'd JSON file without overwriting an existing artifact."""

    payload = _json_bytes(dict(value))
    if path.exists():
        if path.is_symlink() or path.is_junction() or not path.is_file():
            raise LanguageControlCommandError(f"immutable evidence path is unsafe: {path}")
        if path.read_bytes() != payload:
            raise LanguageControlCommandError(f"refusing to overwrite immutable evidence: {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() != payload:
                raise LanguageControlCommandError(
                    f"concurrent immutable evidence differs: {path}"
                ) from error
            return True
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _write_report(path: Path, value: Mapping[str, object]) -> None:
    """Reports are replaceable diagnostics; immutable evidence lives under output/evidence."""

    payload = _json_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _schedule_payload(value: Mapping[str, object]) -> Mapping[str, object]:
    if value.get("schedule_id") is not None:
        return value
    for key in ("control_development", "development"):
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            return cast(Mapping[str, object], candidate)
    raise LanguageControlCommandError("control schedule JSON lacks a development lock")


def load_development_schedule(path: Path) -> ControlScheduleLock:
    """Load exactly the unsealed development lock, rejecting final before materialization."""

    source = _resolved_unlinked(path, label="M5A control schedule")
    payload = _schedule_payload(_read_object(source, label="M5A control schedule"))
    return _parse_development_schedule_payload(payload)


def _parse_development_schedule_payload(
    payload: Mapping[str, object],
) -> ControlScheduleLock:
    """Parse one fingerprint-owned development lock without materializing episodes."""

    if payload.get("schedule_id") != M5A_CONTROL_DEV_SCHEDULE_ID:
        raise LanguageControlCommandError(
            "run_language_control accepts only m5a_control_dev_v0; final remains sealed"
        )
    expected = {
        "schedule_id",
        "split",
        "ordered_scene_seeds",
        "counterpart_scene_seeds",
        "ordered_task_specs",
        "associated_language_example_ids",
        "exclusion_digest",
        "language_schedule_fingerprint",
        "schedule_fingerprint",
        "sealed",
        "schema_version",
        "generation_version",
    }
    if set(payload) != expected:
        raise LanguageControlCommandError("control schedule lock fields differ from schema")
    raw_tasks = payload["ordered_task_specs"]
    if not isinstance(raw_tasks, list) or not all(isinstance(item, Mapping) for item in raw_tasks):
        raise LanguageControlCommandError("control schedule TaskSpecs must be JSON objects")
    try:
        result = ControlScheduleLock(
            schedule_id=cast(str, payload["schedule_id"]),
            split=LanguageSplit(cast(str, payload["split"])),
            ordered_scene_seeds=tuple(cast(list[int], payload["ordered_scene_seeds"])),
            counterpart_scene_seeds=tuple(cast(list[int], payload["counterpart_scene_seeds"])),
            ordered_task_specs=tuple(
                TaskSpec.from_mapping(cast(Mapping[str, object], item)) for item in raw_tasks
            ),
            associated_language_example_ids=tuple(
                cast(list[str], payload["associated_language_example_ids"])
            ),
            exclusion_digest=cast(str, payload["exclusion_digest"]),
            language_schedule_fingerprint=cast(str, payload["language_schedule_fingerprint"]),
            schedule_fingerprint=cast(str, payload["schedule_fingerprint"]),
            sealed=cast(bool, payload["sealed"]),
            schema_version=cast(str, payload["schema_version"]),
            generation_version=cast(str, payload["generation_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LanguageControlCommandError(f"invalid development control lock: {error}") from error
    if result.sealed or result.split is not LanguageSplit.DEVELOPMENT:
        raise LanguageControlCommandError("development control lock cannot be sealed or final")
    return result


def _ordered_unexcluded_seeds(
    config: ControlScheduleConfig,
    *,
    count: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    for offset in range(config.maximum_candidate_count):
        candidate = config.candidate_seed_start + offset
        if candidate > 2**31 - 1:
            break
        if candidate in config.excluded_scene_seeds:
            continue
        selected.append(candidate)
        if len(selected) == count:
            return tuple(selected)
    raise LanguageControlCommandError(
        f"cannot rebuild {count} ordered unexcluded control scene seeds"
    )


def _associated_language_ids(
    *,
    corpus: GeneratedLanguageCorpus,
    split: LanguageSplit,
    scene_count: int,
) -> tuple[str, ...]:
    associated: list[str] = []
    by_task = corpus.routeable_example_ids_by_task[split]
    for scene_index in range(scene_count):
        for task_spec in CANONICAL_TASK_SPECS:
            candidates = by_task.get(stable_task_id(task_spec), ())
            if not candidates:
                raise LanguageControlCommandError(
                    f"language split {split.value} lacks a canonical TaskSpec"
                )
            associated.append(candidates[scene_index % len(candidates)])
    return tuple(associated)


def _rebuild_recorded_control_locks(
    *,
    payload: Mapping[str, object],
    corpus: GeneratedLanguageCorpus,
    config: ControlScheduleConfig,
    language_development: LanguageScheduleLock,
    language_final: LanguageScheduleLock,
) -> tuple[ControlScheduleLock, ControlScheduleLock]:
    raw_development = payload.get("control_development")
    if not isinstance(raw_development, Mapping):
        raise LanguageControlCommandError("control_development must be one schedule lock")
    try:
        recorded = _parse_development_schedule_payload(raw_development)
    except LanguageControlCommandError as error:
        raise LanguageControlCommandError(
            "target-development control_development differs from the rebuilt authoritative lock"
        ) from error
    shape = (len(recorded.ordered_scene_seeds), len(recorded.counterpart_scene_seeds))
    if shape not in {
        (M5A_DEVELOPMENT_SCENE_COUNT, M5A_FINAL_SCENE_COUNT),
        (M5A_LEGACY_DEVELOPMENT_SCENE_COUNT, M5A_LEGACY_FINAL_SCENE_COUNT),
    }:
        raise LanguageControlCommandError("recorded control schedule shape is unauthorized")
    selected = _ordered_unexcluded_seeds(config, count=sum(shape))
    development_seeds = selected[: shape[0]]
    final_seeds = selected[shape[0] :]
    try:
        development = ControlScheduleLock(
            schedule_id=M5A_CONTROL_DEV_SCHEDULE_ID,
            split=LanguageSplit.DEVELOPMENT,
            ordered_scene_seeds=development_seeds,
            counterpart_scene_seeds=final_seeds,
            ordered_task_specs=CANONICAL_TASK_SPECS,
            associated_language_example_ids=_associated_language_ids(
                corpus=corpus,
                split=LanguageSplit.DEVELOPMENT,
                scene_count=shape[0],
            ),
            exclusion_digest=config.exclusion_digest,
            language_schedule_fingerprint=language_development.schedule_fingerprint,
            sealed=False,
        )
        final = ControlScheduleLock(
            schedule_id=M5A_CONTROL_FINAL_SCHEDULE_ID,
            split=LanguageSplit.FINAL,
            ordered_scene_seeds=final_seeds,
            counterpart_scene_seeds=development_seeds,
            ordered_task_specs=CANONICAL_TASK_SPECS,
            associated_language_example_ids=_associated_language_ids(
                corpus=corpus,
                split=LanguageSplit.FINAL,
                scene_count=shape[1],
            ),
            exclusion_digest=config.exclusion_digest,
            language_schedule_fingerprint=language_final.schedule_fingerprint,
            sealed=True,
        )
    except ValueError as error:
        raise LanguageControlCommandError(
            f"cannot rebuild recorded control locks: {error}"
        ) from error
    if recorded != development:
        raise LanguageControlCommandError(
            "target-development control_development differs from the rebuilt authoritative lock"
        )
    return development, final


def _parse_authoritative_control_config(payload: object) -> ControlScheduleConfig:
    if not isinstance(payload, Mapping):
        raise LanguageControlCommandError("control schedule config must be one JSON object")
    expected = {
        "exclusion_sources",
        "candidate_seed_start",
        "maximum_candidate_count",
        "generation_version",
        "schema_version",
        "exclusion_digest",
    }
    if set(payload) != expected:
        raise LanguageControlCommandError("control schedule config fields differ from schema")
    raw_sources = payload.get("exclusion_sources")
    if not isinstance(raw_sources, list):
        raise LanguageControlCommandError("control schedule exclusions must be one JSON list")
    sources: list[SeedExclusionSource] = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, Mapping):
            raise LanguageControlCommandError(
                f"control schedule exclusion {index} must be one JSON object"
            )
        if set(raw_source) != {"source_id", "source_fingerprint", "scene_seeds"}:
            raise LanguageControlCommandError(
                f"control schedule exclusion {index} fields differ from schema"
            )
        raw_seeds = raw_source.get("scene_seeds")
        if not isinstance(raw_seeds, list):
            raise LanguageControlCommandError(
                f"control schedule exclusion {index} scene_seeds must be one JSON list"
            )
        try:
            sources.append(
                SeedExclusionSource(
                    source_id=cast(str, raw_source["source_id"]),
                    source_fingerprint=cast(str, raw_source["source_fingerprint"]),
                    scene_seeds=tuple(cast(list[int], raw_seeds)),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LanguageControlCommandError(
                f"invalid control schedule exclusion {index}: {error}"
            ) from error
    try:
        config = ControlScheduleConfig(
            exclusion_sources=tuple(sources),
            candidate_seed_start=cast(int, payload["candidate_seed_start"]),
            maximum_candidate_count=cast(int, payload["maximum_candidate_count"]),
            generation_version=cast(str, payload["generation_version"]),
            schema_version=cast(str, payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LanguageControlCommandError(f"invalid control schedule config: {error}") from error
    if dict(payload) != config.to_dict():
        raise LanguageControlCommandError(
            "control schedule config or exclusion digest differs from canonical serialization"
        )
    return config


def _sealed_language_lock_payload(
    value: M5AScheduleBundle | LanguageScheduleLock,
) -> dict[str, object]:
    language_final = value.language_final if isinstance(value, M5AScheduleBundle) else value
    return {
        "schedule_id": language_final.schedule_id,
        "schedule_fingerprint": language_final.schedule_fingerprint,
        "corpus_fingerprint": language_final.corpus_fingerprint,
        "split_fingerprint": language_final.split_fingerprint,
        "example_count": len(language_final.ordered_example_ids),
        "sealed": True,
        "texts_materialized": False,
    }


def _sealed_control_lock_payload(
    value: M5AScheduleBundle | ControlScheduleLock,
) -> dict[str, object]:
    control_final = value.control_final if isinstance(value, M5AScheduleBundle) else value
    return {
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
    }


def load_authoritative_development_schedule(
    path: Path,
    *,
    corpus: GeneratedLanguageCorpus,
) -> ControlScheduleLock:
    """Validate and rebuild the complete target-development four-lock authority."""

    source = _resolved_unlinked(path, label="M5A target-development schedule bundle")
    payload = _read_object(source, label="M5A target-development schedule bundle")
    expected_fields = {
        "schema_version",
        "corpus_fingerprint",
        "m43b_evaluation_evidence_fingerprint",
        "control_schedule_config",
        "language_development",
        "language_final",
        "control_development",
        "control_final",
        "all_four_schedules_locked_before_training",
        "language_final_accessed",
        "control_final_accessed",
        "m42_final_accessed",
        "test_split_accessed",
        "historical_fresh_accessed",
        "smolvla_go",
        "final_texts_materialized",
        "final_episodes_materialized",
    }
    if set(payload) != expected_fields:
        raise LanguageControlCommandError(
            "target-development schedule bundle fields differ from the authoritative schema"
        )
    if payload.get("schema_version") != M5A_TARGET_DEVELOPMENT_SCHEDULE_LOCK_SCHEMA:
        raise LanguageControlCommandError("unknown target-development schedule bundle schema")
    if payload.get("corpus_fingerprint") != corpus.manifest.corpus_fingerprint:
        raise LanguageControlCommandError("target-development schedule corpus fingerprint differs")
    m43_fingerprint = payload.get("m43b_evaluation_evidence_fingerprint")
    if (
        not isinstance(m43_fingerprint, str)
        or len(m43_fingerprint.removeprefix("sha256:")) != 64
        or any(
            character not in "0123456789abcdef"
            for character in m43_fingerprint.removeprefix("sha256:")
        )
    ):
        raise LanguageControlCommandError(
            "target-development schedule lacks a valid M4.3b evidence fingerprint"
        )
    expected_flags = {
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
    if any(payload.get(key) is not expected for key, expected in expected_flags.items()):
        raise LanguageControlCommandError(
            "target-development schedule bundle violates final/test/fresh access locks"
        )
    config = _parse_authoritative_control_config(payload.get("control_schedule_config"))
    language_development, language_final = build_language_schedule_locks(corpus)
    control_development, control_final = _rebuild_recorded_control_locks(
        payload=payload,
        corpus=corpus,
        config=config,
        language_development=language_development,
        language_final=language_final,
    )
    expected_locks: dict[str, object] = {
        "language_development": language_development.to_dict(),
        "language_final": _sealed_language_lock_payload(language_final),
        "control_development": control_development.to_dict(),
        "control_final": _sealed_control_lock_payload(control_final),
    }
    for key, expected in expected_locks.items():
        if payload.get(key) != expected:
            raise LanguageControlCommandError(
                f"target-development {key} differs from the rebuilt authoritative lock"
            )
    return control_development


def load_sealed_final_authority(
    path: Path,
    *,
    corpus: GeneratedLanguageCorpus,
) -> dict[str, object]:
    """Return only fingerprints and non-access flags from the validated final locks."""

    load_authoritative_development_schedule(path, corpus=corpus)
    payload = _read_object(
        _resolved_unlinked(path, label="M5A sealed final authority"),
        label="M5A sealed final authority",
    )
    language_final = payload.get("language_final")
    control_final = payload.get("control_final")
    if not isinstance(language_final, Mapping) or not isinstance(control_final, Mapping):
        raise LanguageControlCommandError("sealed final authority lacks both final locks")
    if (
        language_final.get("sealed") is not True
        or language_final.get("texts_materialized") is not False
        or control_final.get("sealed") is not True
        or control_final.get("language_example_ids_materialized") is not False
        or control_final.get("episodes_materialized") is not False
    ):
        raise LanguageControlCommandError("sealed final authority reports forbidden access")
    language_fingerprint = language_final.get("schedule_fingerprint")
    control_fingerprint = control_final.get("schedule_fingerprint")
    if not all(
        isinstance(value, str) and value.startswith("sha256:")
        for value in (language_fingerprint, control_fingerprint)
    ):
        raise LanguageControlCommandError("sealed final lock fingerprint is malformed")
    return {
        "schema_version": "langmani-m5a-sealed-final-authority-v0",
        "final_language_schedule_fingerprint": language_fingerprint,
        "final_control_schedule_fingerprint": control_fingerprint,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "final_command_texts_materialized": False,
        "final_scene_seeds_materialized": False,
    }


def prepare_development_inputs(
    *, schedule: ControlScheduleLock, corpus: GeneratedLanguageCorpus, stage: M5AStage
) -> DevelopmentControlInputs:
    language_development, _language_final = build_language_schedule_locks(corpus)
    if schedule.language_schedule_fingerprint != language_development.schedule_fingerprint:
        raise LanguageControlCommandError("control schedule refers to another language corpus")
    staged = build_staged_control_schedule(schedule, stage=stage)
    episodes = staged.episodes
    expected = M5A_PHYSICAL_STAGE_BUDGETS[stage].oracle_episodes
    if len(episodes) != expected or tuple(item.episode_index for item in episodes) != tuple(
        range(expected)
    ):
        raise LanguageControlCommandError("staged development schedule indices differ")
    examples = corpus.examples_for_split(LanguageSplit.DEVELOPMENT)
    by_id = {example.example_id: example for example in examples}
    for episode in episodes:
        example = by_id.get(episode.language_example_id)
        if example is None:
            raise LanguageControlCommandError("scheduled development command ID is absent")
        if example.expected_task_spec != episode.task_spec:
            raise LanguageControlCommandError(
                "scheduled development command and oracle TaskSpec disagree"
            )
    return DevelopmentControlInputs(
        schedule=staged,
        episodes=episodes,
        examples_by_id=by_id,
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
    )


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LanguageControlCommandError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LanguageControlCommandError(f"{label} must be finite")
    return result


def _classifier_router(
    artifact_root: Path, *, device: str
) -> tuple[FactorizedTextRouterV0, dict[str, object]]:
    artifact_root = _resolved_unlinked(artifact_root, label="classifier artifact root")
    model, tokenizer, manifest = load_factorized_text_classifier(artifact_root)
    evidence = read_text_classifier_run_evidence(artifact_root)
    raw_selection = evidence.get("calibration_selection")
    raw_config = evidence.get("router_config")
    if not isinstance(raw_selection, Mapping) or not isinstance(raw_config, Mapping):
        raise LanguageControlCommandError(
            "classifier run evidence lacks calibration_selection/router_config"
        )
    raw_temperature = raw_selection.get("temperature")
    raw_threshold = raw_selection.get("threshold")
    if not isinstance(raw_temperature, Mapping) or not isinstance(raw_threshold, Mapping):
        raise LanguageControlCommandError("classifier calibration selection is malformed")
    temperature = TemperatureCalibrationV0(
        temperature=_number(raw_temperature.get("temperature"), label="temperature"),
        validation_examples=int(raw_temperature.get("validation_examples", 0)),
        nll_before=_number(raw_temperature.get("nll_before"), label="nll_before"),
        nll_after=_number(raw_temperature.get("nll_after"), label="nll_after"),
        bounded_iterations=int(raw_temperature.get("bounded_iterations", 0)),
    )
    threshold = _number(raw_threshold.get("threshold"), label="routing threshold")
    configured_threshold = _number(
        raw_config.get("routing_threshold"), label="router-config threshold"
    )
    if threshold != configured_threshold:
        raise LanguageControlCommandError(
            "classifier router threshold differs from validation selection"
        )
    maximum_length = raw_config.get("maximum_sequence_length")
    if isinstance(maximum_length, bool) or not isinstance(maximum_length, int):
        raise LanguageControlCommandError("classifier maximum_sequence_length is malformed")
    if raw_config.get("device_independent") is not True:
        raise LanguageControlCommandError(
            "classifier router config must explicitly permit device-independent reload"
        )
    router = FactorizedTextRouterV0(
        model=model,
        tokenizer=cast(object, tokenizer),
        calibration=temperature,
        config=FactorizedTextRouterConfig(
            maximum_sequence_length=maximum_length,
            routing_threshold=threshold,
            device=device,
        ),
    )
    complete = _read_object(artifact_root / "complete.json", label="classifier completion")
    artifact_fingerprint = complete.get("artifact_fingerprint")
    if not isinstance(artifact_fingerprint, str):
        raise LanguageControlCommandError("classifier completion lacks artifact fingerprint")
    identity = {
        "artifact_fingerprint": artifact_fingerprint,
        "architecture_fingerprint": manifest.get("architecture_fingerprint"),
        "run_evidence_fingerprint": f"sha256:{sha256_hex(evidence)}",
        "router_config": {
            "maximum_sequence_length": maximum_length,
            "routing_threshold": threshold,
        },
        "calibration_selection": dict(raw_selection),
    }
    return router, identity


def _load_frozen_router_evaluation(
    evidence_root: Path,
    *,
    corpus: GeneratedLanguageCorpus,
    classifier_identity: Mapping[str, object],
    rule: RuleRouterV0,
    llm_config: StructuredLLMRouterConfig,
    llm_license: str,
    llm_license_reviewed: bool,
) -> tuple[tuple[LanguageExample, ...], dict[str, object]]:
    """Validate and reuse the exact immutable router/prompt selected before control."""

    root = _resolved_unlinked(evidence_root, label="router-evaluation evidence root")
    if not root.is_dir() or root.is_symlink() or root.is_junction():
        raise LanguageControlCommandError(
            "router-evaluation evidence root must be a real directory"
        )
    owner = _read_object(root / "owner.json", label="router-evaluation owner")
    manifest = _read_object(
        root / "artifact_manifest.json", label="router-evaluation artifact manifest"
    )
    completion = _read_object(root / "complete.json", label="router-evaluation completion")
    result = _read_object(root / "result.json", label="router-evaluation result")
    artifacts = _router_evaluation_artifacts(root)
    artifact_fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': artifacts})}"
    if (
        manifest.get("artifacts") != artifacts
        or manifest.get("artifact_fingerprint") != artifact_fingerprint
    ):
        raise LanguageControlCommandError("router-evaluation checksum manifest differs")
    if (
        completion.get("passed") is not True
        or completion.get("artifact_fingerprint") != artifact_fingerprint
        or completion.get("evaluation_fingerprint") != owner.get("evaluation_fingerprint")
        or result.get("passed") is not True
        or result.get("evaluation_fingerprint") != owner.get("evaluation_fingerprint")
    ):
        raise LanguageControlCommandError("router-evaluation completion identity differs")
    if (
        result.get("mode") != "target_development"
        or result.get("target_development_executed") is not True
        or result.get("language_development_completed") is not True
        or result.get("corpus_fingerprint") != corpus.manifest.corpus_fingerprint
    ):
        raise LanguageControlCommandError(
            "control requires completed target language evaluation for the same corpus"
        )
    forbidden = (
        "language_final_accessed",
        "control_final_accessed",
        "m42_final_accessed",
        "test_split_accessed",
        "historical_fresh_accessed",
        "smolvla_go",
    )
    if any(result.get(name) is not False for name in forbidden):
        raise LanguageControlCommandError("router evaluation accessed a forbidden source")
    identities = result.get("router_identities")
    if not isinstance(identities, Mapping) or set(identities) != {"rule", "classifier", "llm"}:
        raise LanguageControlCommandError("router evaluation lacks exactly three identities")
    evaluated_rule = identities["rule"]
    evaluated_classifier = identities["classifier"]
    evaluated_llm = identities["llm"]
    if not all(
        isinstance(value, Mapping)
        for value in (evaluated_rule, evaluated_classifier, evaluated_llm)
    ):
        raise LanguageControlCommandError("router evaluation identities must be objects")
    if evaluated_rule.get("router_fingerprint") != rule.config.fingerprint:
        raise LanguageControlCommandError("control RuleRouter differs from language evaluation")
    if evaluated_classifier.get("artifact_fingerprint") != classifier_identity.get(
        "artifact_fingerprint"
    ) or evaluated_classifier.get("run_evidence_fingerprint") != classifier_identity.get(
        "run_evidence_fingerprint"
    ):
        raise LanguageControlCommandError("control classifier differs from language evaluation")
    evaluated_classifier_config = evaluated_classifier.get("router_config")
    if not isinstance(evaluated_classifier_config, Mapping):
        raise LanguageControlCommandError("evaluated classifier config is malformed")
    if evaluated_classifier_config.get("maximum_sequence_length") != cast(
        Mapping[str, object], classifier_identity["router_config"]
    ).get("maximum_sequence_length") or evaluated_classifier_config.get(
        "routing_threshold"
    ) != cast(Mapping[str, object], classifier_identity["router_config"]).get("routing_threshold"):
        raise LanguageControlCommandError("control classifier runtime differs from evaluation")
    if evaluated_llm.get("config") != llm_config.to_dict():
        raise LanguageControlCommandError("control LLM config differs from language evaluation")
    if (
        evaluated_llm.get("declared_license") != llm_license
        or evaluated_llm.get("official_model_card_license_reviewed") is not True
        or llm_license_reviewed is not True
    ):
        raise LanguageControlCommandError("control LLM license review differs from evaluation")
    prompt_ids = evaluated_llm.get("prompt_example_ids")
    if not isinstance(prompt_ids, list) or not all(isinstance(value, str) for value in prompt_ids):
        raise LanguageControlCommandError("evaluated LLM prompt IDs are malformed")
    canonical = select_structured_routing_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    if prompt_ids != [example.example_id for example in canonical]:
        raise LanguageControlCommandError("evaluated LLM prompt is not the canonical train prompt")
    template, fingerprint, _ = build_structured_routing_prompt(
        examples=canonical,
        prompt_version=llm_config.prompt_version,
    )
    if (
        evaluated_llm.get("prompt_template") != template
        or evaluated_llm.get("prompt_fingerprint") != fingerprint
    ):
        raise LanguageControlCommandError("evaluated LLM prompt content or fingerprint differs")
    promotions = result.get("candidate_promotions")
    promoted = result.get("promoted_learned_routers")
    primary = result.get("primary_learned_router")
    if (
        not isinstance(promotions, Mapping)
        or not isinstance(promoted, list)
        or any(value not in LEARNED_ROUTER_ORDER for value in promoted)
        or len(set(promoted)) != len(promoted)
        or (primary is not None and primary not in promoted)
    ):
        raise LanguageControlCommandError("language evaluation candidate promotions are malformed")
    return canonical, {
        "evaluation_fingerprint": owner.get("evaluation_fingerprint"),
        "artifact_fingerprint": artifact_fingerprint,
        "evidence_root": str(root),
        "router_identities": dict(identities),
        "candidate_promotions": dict(promotions),
        "promoted_learned_routers": list(promoted),
        "primary_learned_router": primary,
    }


def _provider(router: RouterLike) -> DecisionProvider:
    def provide(_episode: ScheduledControlEpisode, example: LanguageExample) -> RouterDecision:
        return router.route(example.raw_text)

    return provide


def _oracle_provider(episode: ScheduledControlEpisode, _example: LanguageExample) -> RouterDecision:
    return RouterDecision.route(
        task_spec=episode.task_spec,
        confidence=RouterConfidence.unavailable(
            definition="oracle TaskSpec is a controller ceiling, not language confidence"
        ),
        router_name="OracleTaskSpecRouterV0",
        router_version="oracle-task-spec-router-v0",
        evidence={"oracle": True},
    )


def _validate_rejection_noop_probe(payload: Mapping[str, object]) -> dict[str, object]:
    probe = dict(payload)
    fingerprint = probe.pop("probe_fingerprint", None)
    if fingerprint != f"sha256:{sha256_hex(probe)}":
        raise LanguageControlCommandError("rejection no-op probe fingerprint differs")
    probe["probe_fingerprint"] = fingerprint
    result = probe.get("result")
    if not isinstance(result, Mapping):
        raise LanguageControlCommandError("rejection no-op probe lacks a dispatch result")
    dispatch = result.get("dispatch")
    decision = result.get("decision")
    if not isinstance(dispatch, Mapping) or not isinstance(decision, Mapping):
        raise LanguageControlCommandError("rejection no-op probe result is malformed")
    if (
        probe.get("schema_version") != REJECTION_NOOP_PROBE_SCHEMA
        or probe.get("physical_m1_environment") is not True
        or probe.get("controller_lookup_count") != 0
        or probe.get("environment_reset_count") != 0
        or probe.get("environment_step_count") != 0
        or decision.get("status") == RouterStatus.ROUTE.value
        or dispatch.get("dispatched") is not False
        or dispatch.get("controller_loaded") is not False
        or dispatch.get("policy_called") is not False
        or dispatch.get("environment_reset_called") is not False
        or dispatch.get("environment_step_count") != 0
        or dispatch.get("safe_rejection") is not True
        or probe.get("passed") is not True
    ):
        raise LanguageControlCommandError("rejection no-op probe entered a forbidden runtime")
    return probe


def run_rejection_noop_probe(
    *,
    registry: ControllerRegistry,
    loader: StrictPerTaskControllerLoader,
    environment: object,
    router: RouterLike,
    command: str = "Open the drawer.",
    evaluation_id: str = "m5a-target-development-rejection-noop-probe",
) -> dict[str, object]:
    """Prove on the active target environment that rejection performs zero work."""
    decision = router.route(command)
    if decision.status is RouterStatus.ROUTE:
        raise LanguageControlCommandError("the rejection probe unexpectedly produced a route")
    counting_loader = _CountingControllerLoader(loader)
    counting_environment = _CountingEnvironmentProxy(environment)
    result = ControllerDispatcher(registry=registry, loader=counting_loader).dispatch(
        decision,
        environment=counting_environment,
        evaluation_id=evaluation_id,
        oracle_task_spec=None,
        scene_seed=None,
    )
    dispatch = result.dispatch
    base: dict[str, object] = {
        "schema_version": REJECTION_NOOP_PROBE_SCHEMA,
        "command_fingerprint": f"sha256:{sha256_hex(command)}",
        "decision_fingerprint": decision.decision_fingerprint,
        "controller_registry_fingerprint": registry.registry_fingerprint,
        "physical_m1_environment": True,
        "controller_lookup_count": counting_loader.lookup_count,
        "environment_reset_count": counting_environment.reset_count,
        "environment_step_count": counting_environment.step_count,
        "result": result.to_dict(),
        "passed": (
            counting_loader.lookup_count == 0
            and counting_environment.reset_count == 0
            and counting_environment.step_count == 0
            and dispatch.dispatched is False
            and dispatch.controller_loaded is False
            and dispatch.policy_called is False
            and dispatch.environment_reset_called is False
            and dispatch.environment_step_count == 0
            and dispatch.safe_rejection is True
        ),
    }
    return _validate_rejection_noop_probe(
        {**base, "probe_fingerprint": f"sha256:{sha256_hex(base)}"}
    )


def run_three_scene_rejection_probes(
    *,
    registry: ControllerRegistry,
    loader: StrictPerTaskControllerLoader,
    environment: object,
    router: RouterLike,
) -> dict[str, object]:
    """Execute the fixed six-command safety set without entering control runtime."""

    items: list[dict[str, object]] = []
    for case in THREE_SCENE_REJECTION_CASES:
        probe = run_rejection_noop_probe(
            registry=registry,
            loader=loader,
            environment=environment,
            router=router,
            command=case.command,
            evaluation_id=f"m5a-three-scene-rejection:{case.probe_id}",
        )
        items.append({**case.to_dict(), "policy_reset_count": 0, "probe": probe})
    return build_rejection_probe_set(items)


def run_full_control_rejection_probes(
    *,
    registry: ControllerRegistry,
    loader: StrictPerTaskControllerLoader,
    environment: object,
    router: RouterLike,
) -> dict[str, object]:
    """Execute the locked nine-command full-development rejection set."""

    items: list[dict[str, object]] = []
    for case in FULL_REJECTION_CASES:
        probe = run_rejection_noop_probe(
            registry=registry,
            loader=loader,
            environment=environment,
            router=router,
            command=case.command,
            evaluation_id=f"m5a-full-control-rejection:{case.probe_id}",
        )
        items.append({**case.to_dict(), "policy_reset_count": 0, "probe": probe})
    return build_full_rejection_probe_set(items)


def _episode_identity(
    *,
    run_fingerprint: str,
    router_label: str,
    schedule: StagedControlSchedule,
    episode: ScheduledControlEpisode,
    example: LanguageExample,
) -> dict[str, object]:
    return {
        "run_fingerprint": run_fingerprint,
        "router_label": router_label,
        "schedule_id": schedule.schedule_id,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "episode_index": episode.episode_index,
        "scene_seed": episode.scene_seed,
        "scene_id": episode.scene_id,
        "oracle_task_id": episode.task_id,
        "language_example_id": example.example_id,
        "command_fingerprint": f"sha256:{sha256_hex(example.raw_text)}",
    }


def _active_episode_metadata(
    environment: object,
    *,
    episode: ScheduledControlEpisode,
    dispatched: bool,
    required: bool,
) -> dict[str, object] | None:
    if not dispatched:
        return None
    base = getattr(environment, "unwrapped", environment)
    accessor = getattr(base, "get_episode_specs", None)
    if not callable(accessor):
        if required:
            raise LanguageControlCommandError("real M1 environment lacks get_episode_specs()")
        return None
    specs = accessor()
    if not isinstance(specs, tuple) or len(specs) != 1:
        raise LanguageControlCommandError("active M1 metadata must contain one EpisodeSpec")
    active = specs[0]
    to_dict = getattr(active, "to_dict", None)
    if not callable(to_dict):
        raise LanguageControlCommandError("active EpisodeSpec lacks JSON serialization")
    payload = to_dict()
    if not isinstance(payload, dict):
        raise LanguageControlCommandError("active EpisodeSpec must serialize to one object")
    if (
        payload.get("scene_seed") != episode.scene_seed
        or payload.get("scene_id") != episode.scene_id
        or payload.get("task_id") != episode.task_id
        or payload.get("task_spec") != episode.task_spec.to_dict()
    ):
        raise LanguageControlCommandError("active EpisodeSpec differs from oracle control schedule")
    return cast(dict[str, object], payload)


def _validated_episode_record(
    path: Path, *, expected_identity: Mapping[str, object]
) -> dict[str, object]:
    record = _read_object(path, label="M5A episode evidence")
    fingerprint = record.pop("record_fingerprint", None)
    expected = f"sha256:{sha256_hex(record)}"
    record["record_fingerprint"] = fingerprint
    if fingerprint != expected:
        raise LanguageControlCommandError("episode evidence fingerprint differs")
    if record.get("schema_version") != EPISODE_SCHEMA:
        raise LanguageControlCommandError("episode evidence schema differs")
    if record.get("identity") != dict(expected_identity):
        raise LanguageControlCommandError("episode evidence belongs to another run atom")
    latency = record.get("router_inference_latency_ms")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, int | float)
        or not math.isfinite(float(latency))
        or float(latency) < 0.0
    ):
        raise LanguageControlCommandError("episode router latency is malformed")
    result = record.get("result")
    if not isinstance(result, Mapping):
        raise LanguageControlCommandError("episode evidence lacks an end-to-end result")
    return record


def _summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    attributions: Counter[str] = Counter()
    dispatched = safe_rejections = successes = correct_routes = 0
    false_rejections = wrong_object_routes = wrong_bin_routes = 0
    wrong_task_routes = wrong_route_episodes = 0
    controller_successes = correct_route_control_successes = 0
    correct_route_control_failures = 0
    wrong_object_in_target_bin = target_in_wrong_bin = target_off_table = timeouts = 0
    wrong_object_interactions = invalid_actions = 0
    malformed_actions = nonfinite_actions = dispatch_after_rejection = 0
    raw_violations = arm_projection = gripper_projection = 0
    inference_latencies: list[float] = []
    environment_latencies: list[float] = []
    router_latencies: list[float] = []
    infrastructure_failures = 0
    for record in records:
        result = record["result"]
        assert isinstance(result, Mapping)
        router_latency = record.get("router_inference_latency_ms")
        if isinstance(router_latency, int | float) and not isinstance(router_latency, bool):
            router_latencies.append(float(router_latency))
        decision = result["decision"]
        dispatch = result["dispatch"]
        assert isinstance(decision, Mapping) and isinstance(dispatch, Mapping)
        dispatched += int(dispatch.get("dispatched") is True)
        safe_rejections += int(dispatch.get("safe_rejection") is True)
        if decision.get("status") != "route":
            dispatch_after_rejection += int(
                dispatch.get("dispatched") is True
                or dispatch.get("controller_loaded") is True
                or dispatch.get("policy_called") is True
                or dispatch.get("environment_reset_called") is True
                or int(dispatch.get("environment_step_count", 0)) > 0
            )
        successes += int(result.get("end_to_end_success") is True)
        route_correct = decision.get("task_id") is not None and decision.get(
            "task_id"
        ) == dispatch.get("oracle_task_id")
        correct_routes += int(route_correct)
        oracle_task = result.get("oracle_task_spec")
        if isinstance(oracle_task, Mapping) and decision.get("status") == "route":
            wrong_object = decision.get("target_object_id") != oracle_task.get("target_object_id")
            wrong_bin = decision.get("target_bin_id") != oracle_task.get("target_bin_id")
            wrong_object_routes += int(wrong_object)
            wrong_bin_routes += int(wrong_bin)
            wrong_task_routes += int(wrong_object and wrong_bin)
            wrong_route_episodes += int(wrong_object or wrong_bin)
        false_rejections += int(
            isinstance(oracle_task, Mapping) and decision.get("status") != "route"
        )
        attribution = result.get("failure_attribution")
        if isinstance(attribution, str):
            attributions[attribution] += 1
        control = result.get("control")
        if not isinstance(control, Mapping):
            infrastructure_failures += int(dispatch.get("controller_load_failed") is True)
            continue
        control_success = control.get("success") is True
        controller_successes += int(control_success)
        correct_route_control_successes += int(route_correct and control_success)
        correct_route_control_failures += int(route_correct and not control_success)
        infrastructure_failures += int(control.get("infrastructure_failed") is True)
        infrastructure_failures += int(control.get("controller_inference_failed") is True)
        timeouts += int(control.get("timeout") is True)
        invalid_actions += int(control.get("invalid_action") is True)
        wrong_object_interactions += int(control.get("wrong_object_interaction") is True)
        final_evaluation = control.get("final_evaluation")
        if isinstance(final_evaluation, Mapping):
            wrong_object_in_target_bin += int(
                final_evaluation.get("wrong_object_in_target_bin") is True
            )
            target_in_wrong_bin += int(final_evaluation.get("target_in_wrong_bin") is True)
            target_off_table += int(final_evaluation.get("target_off_table") is True)
        action = control.get("action_evidence")
        if isinstance(action, Mapping):
            raw_violations += int(action.get("raw_violation_count", 0))
            arm_projection += int(action.get("arm_projected_component_count", 0))
            gripper_projection += int(action.get("gripper_projected_component_count", 0))
            malformed_actions += int(action.get("malformed_action_count", 0))
            nonfinite_actions += int(action.get("nonfinite_action_count", 0))
        latency = control.get("inference_latency_ms")
        if isinstance(latency, list):
            inference_latencies.extend(float(value) for value in latency)
        environment_latency = control.get("environment_latency_ms")
        if isinstance(environment_latency, list):
            environment_latencies.extend(float(value) for value in environment_latency)
    sorted_latency = sorted(inference_latencies)
    sorted_environment_latency = sorted(environment_latencies)
    sorted_router_latency = sorted(router_latencies)

    def percentile(values: Sequence[float], fraction: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, math.ceil(fraction * len(values)) - 1)
        return values[index]

    return {
        "episode_count": len(records),
        "dispatched_count": dispatched,
        "safe_rejection_count": safe_rejections,
        "routing_correct_count": correct_routes,
        "routing_false_rejection_count": false_rejections,
        "routing_wrong_object_count": wrong_object_routes,
        "routing_wrong_bin_count": wrong_bin_routes,
        "routing_wrong_task_count": wrong_task_routes,
        "routing_wrong_episode_count": wrong_route_episodes,
        "controller_task_success_count": controller_successes,
        "routing_correct_control_success_count": correct_route_control_successes,
        "routing_correct_control_failure_count": correct_route_control_failures,
        "end_to_end_success_count": successes,
        "wrong_object_in_target_bin_count": wrong_object_in_target_bin,
        "target_in_wrong_bin_count": target_in_wrong_bin,
        "target_off_table_count": target_off_table,
        "timeout_count": timeouts,
        "wrong_object_interaction_available": True,
        "wrong_object_interaction_count": wrong_object_interactions,
        "invalid_action_count": invalid_actions,
        "malformed_action_count": malformed_actions,
        "nonfinite_action_count": nonfinite_actions,
        "dispatch_after_rejection_count": dispatch_after_rejection,
        "m2_expert_call_count": 0,
        "failure_attribution_counts": {
            value.value: attributions[value.value] for value in FailureAttribution
        },
        "raw_action_violation_count": raw_violations,
        "arm_projected_component_count": arm_projection,
        "gripper_projected_component_count": gripper_projection,
        "inference_latency_ms": {
            "sample_count": len(sorted_latency),
            "p50": percentile(sorted_latency, 0.50),
            "p95": percentile(sorted_latency, 0.95),
            "p99": percentile(sorted_latency, 0.99),
            "total": sum(sorted_latency),
        },
        "environment_latency_ms": {
            "sample_count": len(sorted_environment_latency),
            "p50": percentile(sorted_environment_latency, 0.50),
            "p95": percentile(sorted_environment_latency, 0.95),
            "p99": percentile(sorted_environment_latency, 0.99),
            "total": sum(sorted_environment_latency),
        },
        "router_inference_latency_ms": {
            "sample_count": len(sorted_router_latency),
            "p50": percentile(sorted_router_latency, 0.50),
            "p95": percentile(sorted_router_latency, 0.95),
            "p99": percentile(sorted_router_latency, 0.99),
            "total": sum(sorted_router_latency),
        },
        "observed_end_to_end_runtime_ms": sum(sorted_router_latency)
        + sum(sorted_latency)
        + sum(sorted_environment_latency),
        "infrastructure_failure_count": infrastructure_failures,
    }


def _stage_outcome(
    *,
    stage: M5AStage,
    summaries: Mapping[str, object],
    router_order: tuple[str, ...],
) -> dict[str, object]:
    oracle = cast(Mapping[str, object], summaries["oracle"])
    learned = router_order[1:]
    candidate_checks: dict[str, dict[str, bool]] = {}
    for candidate in learned:
        summary = cast(Mapping[str, object], summaries[candidate])
        common = {
            "routing_error": int(summary["routing_wrong_episode_count"]) == 0,
            "false_rejection": int(summary["routing_false_rejection_count"]) == 0,
            "dispatch_after_rejection": int(summary["dispatch_after_rejection_count"]) == 0,
            "invalid_action": int(summary["invalid_action_count"]) == 0,
            "malformed_action": int(summary["malformed_action_count"]) == 0,
            "nonfinite_action": int(summary["nonfinite_action_count"]) == 0,
            "infrastructure": int(summary["infrastructure_failure_count"]) == 0,
            "m2_expert": int(summary["m2_expert_call_count"]) == 0,
        }
        if stage is M5AStage.THREE_SCENE_CONTROL_SCREEN:
            common.update(
                {
                    "routing_wrong_object": int(summary["routing_wrong_object_count"]) <= 1,
                    "routing_wrong_bin": int(summary["routing_wrong_bin_count"]) <= 1,
                    "routing_error_bound": int(summary["routing_wrong_episode_count"]) <= 1,
                    "false_rejection_bound": int(summary["routing_false_rejection_count"]) <= 1,
                    "target_in_wrong_bin": int(summary["target_in_wrong_bin_count"]) == 0,
                    "target_off_table": int(summary["target_off_table_count"]) == 0,
                    "arm_projection": int(summary["arm_projected_component_count"]) == 0,
                }
            )
            common["routing_error"] = common["routing_error_bound"]
            common["false_rejection"] = common["false_rejection_bound"]
        elif stage is M5AStage.FULL_CONTROL_DEVELOPMENT:
            oracle_success = int(oracle["end_to_end_success_count"])
            predicted_success = int(summary["end_to_end_success_count"])
            common.update(
                {
                    "success_gap": abs(oracle_success - predicted_success) <= 2,
                    "routing_correct_count_at_least_35": (
                        int(summary["routing_correct_count"]) >= 35
                    ),
                    "routing_wrong_object": int(summary["routing_wrong_object_count"]) <= 2,
                    "routing_wrong_bin": int(summary["routing_wrong_bin_count"]) <= 2,
                    "false_rejection_bound": int(summary["routing_false_rejection_count"]) <= 3,
                    "target_in_wrong_bin": int(summary["target_in_wrong_bin_count"]) == 0,
                    "target_off_table": int(summary["target_off_table_count"]) == 0,
                    "arm_projection": int(summary["arm_projected_component_count"]) == 0,
                }
            )
            common["routing_error"] = common["routing_correct_count_at_least_35"]
            common["false_rejection"] = common["false_rejection_bound"]
        candidate_checks[candidate] = common
    passed_candidates = tuple(
        candidate for candidate in learned if all(candidate_checks[candidate].values())
    )
    selected: str | None = None
    if stage is M5AStage.THREE_SCENE_CONTROL_SCREEN and passed_candidates:
        ranked: list[tuple[tuple[float, ...], str]] = []
        oracle_success = int(oracle["end_to_end_success_count"])
        for candidate in passed_candidates:
            summary = cast(Mapping[str, object], summaries[candidate])
            latency = cast(Mapping[str, object], summary["router_inference_latency_ms"])
            ranked.append(
                (
                    (
                        abs(oracle_success - int(summary["end_to_end_success_count"])),
                        int(summary["routing_wrong_episode_count"]),
                        int(summary["routing_false_rejection_count"]),
                        int(summary["wrong_object_interaction_count"]),
                        float(latency["p95"] or 0.0),
                    ),
                    candidate,
                )
            )
        ranked.sort()
        selected = ranked[0][1]
    elif stage is M5AStage.FULL_CONTROL_DEVELOPMENT and len(learned) == 1:
        selected = learned[0]
    return {
        "candidate_gate_checks": candidate_checks,
        "promoted_to_next_stage": list(passed_candidates),
        "selected_router": selected,
        "stage_quality_gate_passed": (len(passed_candidates) == len(learned) and bool(learned)),
        "development_quality_gate_passed": (
            stage is M5AStage.FULL_CONTROL_DEVELOPMENT and len(passed_candidates) == 1
        ),
        "final_benchmark_authorized": (
            stage is M5AStage.FULL_CONTROL_DEVELOPMENT and len(passed_candidates) == 1
        ),
    }


def _validate_completed_run(
    *,
    completion_path: Path,
    run_root: Path,
    run_fingerprint: str,
    inputs: DevelopmentControlInputs,
    registry: ControllerRegistry,
    rejection_noop_probe: Mapping[str, object] | None,
    router_order: tuple[str, ...],
) -> dict[str, object]:
    completed = _read_object(completion_path, label="M5A control completion")
    declared_fingerprint = completed.pop("completion_fingerprint", None)
    expected_fingerprint = f"sha256:{sha256_hex(completed)}"
    completed["completion_fingerprint"] = declared_fingerprint
    if declared_fingerprint != expected_fingerprint:
        raise LanguageControlCommandError("completed control evidence fingerprint differs")
    if (
        completed.get("schema_version") != COMPLETION_SCHEMA
        or completed.get("run_fingerprint") != run_fingerprint
        or completed.get("passed") is not True
        or completed.get("completed_episode_atoms") != len(router_order) * len(inputs.episodes)
        or completed.get("expected_episode_atoms") != len(router_order) * len(inputs.episodes)
    ):
        raise LanguageControlCommandError("completed control run identity or counts differ")
    if (
        _read_object(run_root / "controller_registry.json", label="immutable controller registry")
        != registry.to_dict()
    ):
        raise LanguageControlCommandError("completed controller registry artifact differs")
    if rejection_noop_probe is not None:
        stored_probe = _validate_rejection_noop_probe(
            _read_object(
                run_root / "rejection_noop_probe.json",
                label="rejection no-op probe",
            )
        )
        if stored_probe != dict(rejection_noop_probe):
            raise LanguageControlCommandError("completed rejection no-op probe differs")
        if completed.get("rejection_noop_probe_fingerprint") != stored_probe.get(
            "probe_fingerprint"
        ):
            raise LanguageControlCommandError("completion omits the rejection no-op probe")
        if (
            completed.get("rejection_noop_probe_validated") is not True
            or completed.get("zero_dispatch_after_rejection_validated") is not True
        ):
            raise LanguageControlCommandError("completion did not validate rejection no-op")
    summaries: dict[str, object] = {}
    record_fingerprints: list[str] = []
    for router_label in router_order:
        records: list[dict[str, object]] = []
        for episode in inputs.episodes:
            example = inputs.examples_by_id[episode.language_example_id]
            identity = _episode_identity(
                run_fingerprint=run_fingerprint,
                router_label=router_label,
                schedule=inputs.schedule,
                episode=episode,
                example=example,
            )
            record = _validated_episode_record(
                run_root / "episodes" / router_label / f"{episode.episode_index:03d}.json",
                expected_identity=identity,
            )
            records.append(record)
            record_fingerprint = record.get("record_fingerprint")
            assert isinstance(record_fingerprint, str)
            record_fingerprints.append(record_fingerprint)
        summaries[router_label] = _summary(records)
    if completed.get("summaries") != summaries:
        raise LanguageControlCommandError("completed control summaries differ from episodes")
    outcome = _stage_outcome(
        stage=inputs.schedule.stage,
        summaries=summaries,
        router_order=router_order,
    )
    if any(completed.get(key) != value for key, value in outcome.items()):
        raise LanguageControlCommandError("completed stage promotion outcome differs")
    if completed.get("record_set_fingerprint") != (f"sha256:{sha256_hex(record_fingerprints)}"):
        raise LanguageControlCommandError("completed episode-set fingerprint differs")
    return completed


def run_development_control(
    *,
    inputs: DevelopmentControlInputs,
    registry: ControllerRegistry,
    dispatcher: ControllerDispatcher,
    environment: object,
    providers: Mapping[str, DecisionProvider],
    output_root: Path,
    run_identity: Mapping[str, object],
    require_active_episode_spec: bool = False,
    rejection_noop_probe: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run or resume exactly the Oracle and candidates promoted to this stage."""

    router_order = tuple(providers)
    if not router_order or router_order[0] != "oracle":
        raise LanguageControlCommandError("every physical stage must run Oracle first")
    learned = router_order[1:]
    if any(value not in LEARNED_ROUTER_ORDER for value in learned) or len(set(learned)) != len(
        learned
    ):
        raise LanguageControlCommandError("physical stages accept only promoted learned routers")
    budget = M5A_PHYSICAL_STAGE_BUDGETS[inputs.schedule.stage]
    if len(learned) > budget.maximum_learned_candidates:
        raise LanguageControlCommandError("physical stage exceeds its learned-candidate budget")
    if inputs.schedule.stage is M5AStage.FULL_CONTROL_DEVELOPMENT and len(learned) != 1:
        raise LanguageControlCommandError("full development requires one selected learned router")
    validated_probe = (
        None
        if rejection_noop_probe is None
        else _validate_rejection_noop_probe(rejection_noop_probe)
    )
    identity = dict(run_identity)
    identity.update(
        {
            "schema_version": RUN_SCHEMA,
            "schedule_fingerprint": inputs.schedule.schedule_fingerprint,
            "corpus_fingerprint": inputs.corpus_fingerprint,
            "controller_registry_fingerprint": registry.registry_fingerprint,
            "stage": inputs.schedule.stage.value,
            "router_order": list(router_order),
            "episode_count_per_router": len(inputs.episodes),
            "active_episode_spec_required": require_active_episode_spec,
            "rejection_noop_probe_fingerprint": (
                None if validated_probe is None else validated_probe["probe_fingerprint"]
            ),
        }
    )
    run_fingerprint = f"sha256:{sha256_hex(identity)}"
    evidence_root = _resolved_unlinked(output_root / "evidence", label="control evidence root")
    run_root = _resolved_unlinked(
        evidence_root / run_fingerprint.removeprefix("sha256:"),
        label="control run root",
    )
    if run_root.parent != evidence_root:
        raise LanguageControlCommandError("control run escaped its evidence root")
    run_root.mkdir(parents=True, exist_ok=True)
    if run_root.is_symlink() or run_root.is_junction():
        raise LanguageControlCommandError("control run root cannot be linked")
    owner = {
        "schema_version": RUN_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "identity": identity,
    }
    _write_immutable_json(run_root / "owner.json", owner)
    _write_immutable_json(run_root / "controller_registry.json", registry.to_dict())
    if validated_probe is not None:
        _write_immutable_json(run_root / "rejection_noop_probe.json", validated_probe)
    completion_path = run_root / "complete.json"
    if completion_path.exists():
        completed = _validate_completed_run(
            completion_path=completion_path,
            run_root=run_root,
            run_fingerprint=run_fingerprint,
            inputs=inputs,
            registry=registry,
            rejection_noop_probe=validated_probe,
            router_order=router_order,
        )
        return {**completed, "evidence_root": str(run_root), "reused": True}

    all_summaries: dict[str, object] = {}
    all_record_fingerprints: list[str] = []
    completed_atoms = 0
    records_by_router: dict[str, dict[int, dict[str, object]]] = {
        label: {} for label in router_order
    }
    paired_environment = (
        _PairedInitialStateEnvironment(environment)
        if inputs.schedule.stage
        in {M5AStage.THREE_SCENE_CONTROL_SCREEN, M5AStage.FULL_CONTROL_DEVELOPMENT}
        else None
    )
    runtime_environment = environment if paired_environment is None else paired_environment
    atom_order = (
        ((router_label, episode) for episode in inputs.episodes for router_label in router_order)
        if paired_environment is not None
        else (
            (router_label, episode) for router_label in router_order for episode in inputs.episodes
        )
    )
    for router_label, episode in atom_order:
        provider = providers[router_label]
        example = inputs.examples_by_id[episode.language_example_id]
        expected_identity = _episode_identity(
            run_fingerprint=run_fingerprint,
            router_label=router_label,
            schedule=inputs.schedule,
            episode=episode,
            example=example,
        )
        path = run_root / "episodes" / router_label / f"{episode.episode_index:03d}.json"
        if path.exists():
            record = _validated_episode_record(path, expected_identity=expected_identity)
        else:
            if paired_environment is not None:
                paired_environment.begin_episode()
            started_at = time.perf_counter()
            decision = provider(episode, example)
            router_latency_ms = (time.perf_counter() - started_at) * 1_000.0
            result = dispatcher.dispatch(
                decision,
                environment=runtime_environment,
                evaluation_id=(
                    f"{inputs.schedule.schedule_id}:{router_label}:{episode.episode_index:03d}"
                ),
                oracle_task_spec=episode.task_spec,
                scene_seed=episode.scene_seed,
            )
            active_episode_spec = _active_episode_metadata(
                runtime_environment,
                episode=episode,
                dispatched=result.dispatch.dispatched,
                required=require_active_episode_spec,
            )
            base_record: dict[str, object] = {
                "schema_version": EPISODE_SCHEMA,
                "identity": expected_identity,
                "router_inference_latency_ms": router_latency_ms,
                "active_episode_spec": active_episode_spec,
                "result": result.to_dict(),
            }
            if paired_environment is not None:
                initial_audit = paired_environment.consume(dispatched=result.dispatch.dispatched)
                episode_audit = dispatcher.last_episode_audit
                if result.dispatch.dispatched and episode_audit is None:
                    raise LanguageControlCommandError(
                        "paired control execution omitted the policy reset and rollout audit"
                    )
                base_record["paired_execution_audit"] = (
                    None
                    if initial_audit is None
                    else {**initial_audit, **dict(episode_audit or {})}
                )
            record = {
                **base_record,
                "record_fingerprint": f"sha256:{sha256_hex(base_record)}",
            }
            _write_immutable_json(path, record)
            record = _validated_episode_record(path, expected_identity=expected_identity)
        records_by_router[router_label][episode.episode_index] = record
        completed_atoms += 1

    for router_label in router_order:
        records = [
            records_by_router[router_label][episode.episode_index] for episode in inputs.episodes
        ]
        all_summaries[router_label] = _summary(records)
        for record in records:
            fingerprint = record.get("record_fingerprint")
            assert isinstance(fingerprint, str)
            all_record_fingerprints.append(fingerprint)

    integrity_failures = sum(
        int(cast(Mapping[str, object], value)["infrastructure_failure_count"])
        for value in all_summaries.values()
    )
    if integrity_failures:
        raise LanguageControlCommandError(
            f"control evidence contains {integrity_failures} infrastructure failures"
        )
    dispatch_after_rejection_count = sum(
        int(cast(Mapping[str, object], value)["dispatch_after_rejection_count"])
        for value in all_summaries.values()
    )
    if dispatch_after_rejection_count:
        raise LanguageControlCommandError(
            "a rejected router decision entered the controller/environment runtime"
        )
    observed_safe_rejection_count = sum(
        int(cast(Mapping[str, object], value)["safe_rejection_count"])
        for value in all_summaries.values()
    )
    zero_dispatch_validated = dispatch_after_rejection_count == 0 and (
        validated_probe is not None or observed_safe_rejection_count > 0
    )
    stage_outcome = _stage_outcome(
        stage=inputs.schedule.stage,
        summaries=all_summaries,
        router_order=router_order,
    )
    completion_base: dict[str, object] = {
        "schema_version": COMPLETION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "schedule_id": inputs.schedule.schedule_id,
        "schedule_fingerprint": inputs.schedule.schedule_fingerprint,
        "corpus_fingerprint": inputs.corpus_fingerprint,
        "controller_registry_fingerprint": registry.registry_fingerprint,
        "stage": inputs.schedule.stage.value,
        "router_order": list(router_order),
        "completed_episode_atoms": completed_atoms,
        "expected_episode_atoms": len(router_order) * len(inputs.episodes),
        "summaries": all_summaries,
        "record_set_fingerprint": f"sha256:{sha256_hex(all_record_fingerprints)}",
        "wrong_object_interaction_available": True,
        "wrong_object_interaction_count": sum(
            int(cast(Mapping[str, object], value)["wrong_object_interaction_count"])
            for value in all_summaries.values()
        ),
        "invalid_action_count": sum(
            int(cast(Mapping[str, object], value)["invalid_action_count"])
            for value in all_summaries.values()
        ),
        "malformed_action_count": sum(
            int(cast(Mapping[str, object], value)["malformed_action_count"])
            for value in all_summaries.values()
        ),
        "nonfinite_action_count": sum(
            int(cast(Mapping[str, object], value)["nonfinite_action_count"])
            for value in all_summaries.values()
        ),
        "dispatch_after_rejection_count": dispatch_after_rejection_count,
        "rejection_noop_probe_fingerprint": (
            None if validated_probe is None else validated_probe["probe_fingerprint"]
        ),
        "rejection_noop_probe_validated": validated_probe is not None,
        "zero_dispatch_after_rejection_validated": zero_dispatch_validated,
        "m2_expert_call_count": 0,
        "m2_expert_free_validated": True,
        **stage_outcome,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "smolvla_go": False,
        "passed": completed_atoms == len(router_order) * len(inputs.episodes),
    }
    completion = {
        **completion_base,
        "completion_fingerprint": f"sha256:{sha256_hex(completion_base)}",
    }
    _write_immutable_json(completion_path, completion)
    return {**completion, "evidence_root": str(run_root), "reused": False}


def _finalize_three_scene_screen(
    *,
    output_root: Path,
    inputs: DevelopmentControlInputs,
    registry: ControllerRegistry,
    control_result: Mapping[str, object],
    rejection_probe_set: Mapping[str, object],
    prior_authority: Mapping[str, object],
    dispatch_source: Mapping[str, object],
    controller_binding: Mapping[str, object],
    git_commit: str,
) -> dict[str, object]:
    """Build the compact paired screen archive after all 36 atoms are closed."""

    evidence_value = control_result.get("evidence_root")
    if not isinstance(evidence_value, str):
        raise LanguageControlCommandError("paired control result omitted its atom evidence root")
    atom_root = _resolved_unlinked(Path(evidence_value), label="paired control atom evidence")
    oracle_records = [
        _read_object(atom_root / "episodes" / "oracle" / f"{index:03d}.json", label="Oracle atom")
        for index in range(18)
    ]
    learned_records = [
        _read_object(
            atom_root / "episodes" / NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL / f"{index:03d}.json",
            label="NeuroSymbolic atom",
        )
        for index in range(18)
    ]
    analysis = analyze_three_scene_records(
        oracle_records=oracle_records,
        neuro_symbolic_records=learned_records,
        rejection_probe_set=rejection_probe_set,
    )
    identity = {
        "schema_version": "langmani-m5a-three-scene-owner-v0",
        "implementation_git_commit": git_commit,
        "schedule_fingerprint": inputs.schedule.schedule_fingerprint,
        "corpus_fingerprint": inputs.corpus_fingerprint,
        "controller_registry_fingerprint": registry.registry_fingerprint,
        "dispatch_source_fingerprint": dispatch_source.get("dispatch_source_fingerprint"),
        "controller_binding_fingerprint": controller_binding.get("binding_fingerprint"),
        "prior_one_scene_completion_fingerprint": prior_authority.get("completion_fingerprint"),
        "control_run_fingerprint": control_result.get("run_fingerprint"),
        "control_completion_fingerprint": control_result.get("completion_fingerprint"),
        "analysis_fingerprint": analysis["analysis_fingerprint"],
    }
    run_fingerprint = f"sha256:{sha256_hex(identity)}"
    owner = {**identity, "run_fingerprint": run_fingerprint}
    schedule_artifact = {
        **inputs.schedule.to_dict(),
        "commands": [
            {
                "episode_index": episode.episode_index,
                "language_example_id": episode.language_example_id,
                "command": inputs.examples_by_id[episode.language_example_id].raw_text,
            }
            for episode in inputs.episodes
        ],
    }
    flags = {
        "prior_one_scene_evidence_validated": True,
        "neuro_symbolic_router_identity_validated": True,
        "controller_registry_validated": True,
        "three_scene_schedule_validated": True,
        "paired_initial_states_validated": analysis["paired_initial_state_count"] == 18,
        "oracle_control_executed": True,
        "neuro_symbolic_control_executed": True,
        "six_tasks_per_scene_validated": True,
        "routing_results_validated": True,
        "rejection_no_dispatch_validated": True,
        "failure_attribution_validated": True,
        "paired_comparison_validated": True,
        "action_runtime_validated": True,
        "three_scene_control_screen_completed": True,
        "three_scene_control_screen_passed": bool(analysis["three_scene_control_screen_passed"]),
        "selected_router_locked": bool(analysis["selected_router_locked"]),
        "full_control_development_authorized": bool(
            analysis["full_control_development_authorized"]
        ),
        "full_control_development_completed": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "physical_target_validated": True,
    }
    summary_text = (
        "# M5A Three-Scene Paired Control Screen\n\n"
        f"Oracle success: {analysis['oracle_success_count']}/18.\n\n"
        f"NeuroSymbolic success: {analysis['neuro_symbolic_success_count']}/18.\n\n"
        f"Quality screen passed: {str(analysis['three_scene_control_screen_passed']).lower()}.\n\n"
        "Full control development and all final/test/fresh schedules were not accessed.\n"
    )
    artifacts: dict[str, object] = {
        "input_contract.json": {
            "stage": M5AStage.THREE_SCENE_CONTROL_SCREEN.value,
            "episode_pairs": 18,
            "oracle_episodes": 18,
            "neuro_symbolic_episodes": 18,
            "prior_one_scene_authority": dict(prior_authority),
            "prohibited_sources_accessed": False,
        },
        "schedule.json": schedule_artifact,
        "router_identity.json": dict(dispatch_source),
        "controller_registry.json": registry.to_dict(),
        "runtime_identity.json": {
            "control_atom_evidence_root": str(atom_root),
            "control_run_fingerprint": control_result.get("run_fingerprint"),
            "control_completion_fingerprint": control_result.get("completion_fingerprint"),
            "controller_binding": dict(controller_binding),
            "control_mode": "pd_joint_pos",
            "execution_horizon": 10,
            "action_bound_mode": "project",
            "policy_reset_per_episode": True,
            "m2_expert_call_count": 0,
        },
        "oracle_episodes.json": {"episodes": oracle_records},
        "neuro_symbolic_episodes.json": {"episodes": learned_records},
        "paired_results.json": {
            "pairs": analysis["paired_results"],
            "paired_outcome_counts": analysis["paired_outcome_counts"],
        },
        "rejection_probes.json": dict(rejection_probe_set),
        "failure_attribution.json": {
            "counts": analysis["failure_attribution_counts"],
            "one_primary_category_per_neuro_symbolic_episode": True,
        },
        "benchmark_summary.json": analysis,
        "gate_result.json": {
            "gate_items": analysis["gate_items"],
            "three_scene_control_screen_passed": analysis["three_scene_control_screen_passed"],
            "selected_router_locked": analysis["selected_router_locked"],
            "full_control_development_authorized": analysis["full_control_development_authorized"],
            "full_control_development_completed": False,
        },
        "summary.md": summary_text,
    }
    evidence = write_three_scene_evidence(
        output_root / "three-scene-control-screen",
        owner=owner,
        artifacts=artifacts,
        flags=flags,
    )
    complete = cast(Mapping[str, object], evidence["complete"])
    return {
        "three_scene_evidence_root": evidence["root"],
        "three_scene_run_fingerprint": run_fingerprint,
        "three_scene_artifact_fingerprint": evidence["artifact_fingerprint"],
        "three_scene_completion_fingerprint": complete["completion_fingerprint"],
        "three_scene_evidence_reused": evidence["evidence_reused"],
        "paired_analysis": analysis,
        "stage_quality_gate_passed": bool(analysis["three_scene_control_screen_passed"]),
        "promoted_to_next_stage": (
            [NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL]
            if analysis["three_scene_control_screen_passed"]
            else []
        ),
        "selected_router": (
            NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL
            if analysis["three_scene_control_screen_passed"]
            else None
        ),
        **flags,
        "passed": True,
    }


def _finalize_full_control_development(
    *,
    output_root: Path,
    inputs: DevelopmentControlInputs,
    registry: ControllerRegistry,
    control_result: Mapping[str, object],
    rejection_probe_set: Mapping[str, object],
    prior_authority: Mapping[str, object],
    sealed_final_authority: Mapping[str, object],
    dispatch_source: Mapping[str, object],
    controller_binding: Mapping[str, object],
    git_commit: str,
) -> dict[str, object]:
    """Build the immutable six-scene archive and bounded final authorization."""

    evidence_value = control_result.get("evidence_root")
    if not isinstance(evidence_value, str):
        raise LanguageControlCommandError("full control result omitted its atom evidence root")
    atom_root = _resolved_unlinked(Path(evidence_value), label="full control atom evidence")
    oracle_records = [
        _read_object(atom_root / "episodes" / "oracle" / f"{index:03d}.json", label="Oracle atom")
        for index in range(36)
    ]
    learned_records = [
        _read_object(
            atom_root / "episodes" / NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL / f"{index:03d}.json",
            label="NeuroSymbolic atom",
        )
        for index in range(36)
    ]
    summaries = control_result.get("summaries")
    if not isinstance(summaries, Mapping):
        raise LanguageControlCommandError("full control result omitted its router summaries")
    analysis = analyze_full_control_records(
        oracle_records=oracle_records,
        neuro_symbolic_records=learned_records,
        rejection_probe_set=rejection_probe_set,
        examples_by_id=inputs.examples_by_id,
        router_summaries=summaries,
    )
    runtime_base = {
        "controller_binding": dict(controller_binding),
        "control_mode": "pd_joint_pos",
        "execution_horizon": 10,
        "action_bound_mode": "project",
        "policy_reset_per_episode": True,
        "m2_expert_call_count": 0,
    }
    runtime_fingerprint = f"sha256:{sha256_hex(runtime_base)}"
    identity = {
        "schema_version": "langmani-m5a-full-control-owner-v0",
        "implementation_git_commit": git_commit,
        "schedule_fingerprint": inputs.schedule.schedule_fingerprint,
        "corpus_fingerprint": inputs.corpus_fingerprint,
        "controller_registry_fingerprint": registry.registry_fingerprint,
        "dispatch_source_fingerprint": dispatch_source.get("dispatch_source_fingerprint"),
        "controller_binding_fingerprint": controller_binding.get("binding_fingerprint"),
        "prior_three_scene_completion_fingerprint": prior_authority.get("completion_fingerprint"),
        "control_run_fingerprint": control_result.get("run_fingerprint"),
        "control_completion_fingerprint": control_result.get("completion_fingerprint"),
        "analysis_fingerprint": analysis["analysis_fingerprint"],
        "runtime_contract_fingerprint": runtime_fingerprint,
    }
    run_fingerprint = f"sha256:{sha256_hex(identity)}"
    owner = {**identity, "run_fingerprint": run_fingerprint}
    quality_passed = analysis["development_quality_gate_passed"] is True
    final_authorization = build_final_authorization(
        authorized=quality_passed,
        router_fingerprint=cast(str, dispatch_source["router_lock_fingerprint"]),
        controller_registry_fingerprint=registry.registry_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        full_schedule_fingerprint=inputs.schedule.schedule_fingerprint,
        development_evidence_fingerprint=run_fingerprint,
        selected_router_identity="NeuroSymbolicRouterV0",
        final_language_schedule_fingerprint=cast(
            str, sealed_final_authority["final_language_schedule_fingerprint"]
        ),
        final_control_schedule_fingerprint=cast(
            str, sealed_final_authority["final_control_schedule_fingerprint"]
        ),
        git_commit=git_commit,
    )
    flags = {
        "prior_one_scene_evidence_validated": True,
        "prior_three_scene_evidence_validated": True,
        "selected_router_identity_validated": True,
        "controller_registry_validated": True,
        "full_development_schedule_validated": True,
        "paired_initial_states_validated": analysis["paired_initial_state_count"] == 36,
        "oracle_control_development_completed": True,
        "neuro_symbolic_control_development_completed": True,
        "routing_results_validated": True,
        "rejection_no_dispatch_validated": True,
        "failure_attribution_validated": True,
        "paired_comparison_validated": True,
        "action_runtime_validated": True,
        "full_control_development_completed": True,
        "development_quality_gate_passed": quality_passed,
        "selected_router_locked": quality_passed,
        "final_benchmark_authorized": quality_passed,
        "final_authorization_created": quality_passed,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "physical_target_validated": True,
    }
    schedule_artifact = {
        **inputs.schedule.to_dict(),
        "commands": [
            {
                "episode_index": episode.episode_index,
                "language_example_id": episode.language_example_id,
                "command": inputs.examples_by_id[episode.language_example_id].raw_text,
            }
            for episode in inputs.episodes
        ],
    }
    summary_text = (
        "# M5A Full Six-Scene Paired Control Development\n\n"
        f"Oracle success: {analysis['oracle_success_count']}/36.\n\n"
        f"NeuroSymbolic success: {analysis['neuro_symbolic_success_count']}/36.\n\n"
        f"Correct routes: {analysis['routing_correct_count']}/36.\n\n"
        f"Quality gate passed: {str(quality_passed).lower()}.\n\n"
        "No final, test, historical-fresh, m42-final, M2, or SmolVLA source was accessed.\n"
    )
    artifacts: dict[str, object] = {
        "input_contract.json": {
            "stage": M5AStage.FULL_CONTROL_DEVELOPMENT.value,
            "episode_pairs": 36,
            "oracle_episodes": 36,
            "neuro_symbolic_episodes": 36,
            "prior_three_scene_authority": dict(prior_authority),
            "sealed_final_authority": dict(sealed_final_authority),
            "prohibited_sources_accessed": False,
        },
        "schedule.json": schedule_artifact,
        "router_identity.json": dict(dispatch_source),
        "controller_registry.json": registry.to_dict(),
        "runtime_identity.json": {
            **runtime_base,
            "runtime_contract_fingerprint": runtime_fingerprint,
            "control_atom_evidence_root": str(atom_root),
            "control_run_fingerprint": control_result.get("run_fingerprint"),
            "control_completion_fingerprint": control_result.get("completion_fingerprint"),
        },
        "oracle_episodes.json": {"episodes": oracle_records},
        "neuro_symbolic_episodes.json": {"episodes": learned_records},
        "paired_results.json": {
            "pairs": analysis["paired_results"],
            "paired_outcome_counts": analysis["paired_outcome_counts"],
        },
        "rejection_probes.json": dict(rejection_probe_set),
        "failure_attribution.json": {
            "counts": analysis["failure_attribution_counts"],
            "one_primary_category_per_neuro_symbolic_episode": True,
        },
        "benchmark_summary.json": analysis,
        "gate_result.json": {
            "gate_items": analysis["gate_items"],
            "development_quality_gate_passed": quality_passed,
            "selected_router_locked": quality_passed,
            "final_benchmark_authorized": quality_passed,
        },
        "final_authorization.json": final_authorization,
        "summary.md": summary_text,
    }
    evidence = write_full_control_evidence(
        output_root / "full-control-development",
        owner=owner,
        artifacts=artifacts,
        flags=flags,
    )
    complete = cast(Mapping[str, object], evidence["complete"])
    return {
        "full_control_evidence_root": evidence["root"],
        "full_control_run_fingerprint": run_fingerprint,
        "full_control_artifact_fingerprint": evidence["artifact_fingerprint"],
        "full_control_completion_fingerprint": complete["completion_fingerprint"],
        "full_control_evidence_reused": evidence["evidence_reused"],
        "paired_analysis": analysis,
        "final_authorization": final_authorization,
        "stage_quality_gate_passed": quality_passed,
        "promoted_to_next_stage": [],
        "selected_router": NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL if quality_passed else None,
        **flags,
        "passed": True,
    }


def _create_environment() -> object:
    import gymnasium as gym

    import langmani.environments  # noqa: F401 - registers the one M1 environment
    from langmani.environments.pick_place_by_instruction import ENV_ID

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )


def _load_frozen_controller_registry(
    *,
    checkpoint_root: Path,
    dataset_root: Path,
    runtime_selection_path: Path,
) -> tuple[ControllerRegistry, ControllerRegistryLocators]:
    """Load only completion, validation-selection, and frozen runtime metadata."""

    return load_controller_registry_metadata(
        checkpoint_root=checkpoint_root,
        dataset_root=dataset_root,
        runtime_selection_path=runtime_selection_path,
    )


def _stage_candidates(
    *,
    stage: M5AStage,
    router_evaluation_identity: Mapping[str, object],
    prior_stage_report: Path | None,
) -> tuple[str, ...]:
    promoted = router_evaluation_identity.get("promoted_learned_routers")
    primary = router_evaluation_identity.get("primary_learned_router")
    if not isinstance(promoted, list) or any(
        value not in LEARNED_ROUTER_ORDER for value in promoted
    ):
        raise LanguageControlCommandError("language promotion list is malformed")
    if stage is M5AStage.ONE_SCENE_CONTROL_SMOKE:
        if primary is None:
            return ()
        if primary not in promoted:
            raise LanguageControlCommandError("primary router was not language-promoted")
        return (cast(str, primary), *(value for value in promoted if value != primary))
    if prior_stage_report is None:
        raise LanguageControlCommandError("later physical stage requires prior-stage evidence")
    prior = _read_object(
        _resolved_unlinked(prior_stage_report, label="prior M5A control stage report"),
        label="prior M5A control stage report",
    )
    if prior.get("passed") is not True or prior.get("control_final_accessed") is not False:
        raise LanguageControlCommandError("prior physical stage is incomplete or accessed final")
    if stage is M5AStage.THREE_SCENE_CONTROL_SCREEN:
        if prior.get("stage") != M5AStage.ONE_SCENE_CONTROL_SMOKE.value:
            raise LanguageControlCommandError(
                "three-scene screen requires one-scene smoke evidence"
            )
        values = prior.get("promoted_to_next_stage")
        if not isinstance(values, list) or any(value not in promoted for value in values):
            raise LanguageControlCommandError("one-scene promotion evidence is malformed")
        return tuple(cast(list[str], values))
    if prior.get("stage") != M5AStage.THREE_SCENE_CONTROL_SCREEN.value:
        raise LanguageControlCommandError("full development requires three-scene screen evidence")
    selected = prior.get("selected_router")
    candidates = prior.get("promoted_to_next_stage")
    if (
        selected not in LEARNED_ROUTER_ORDER
        or not isinstance(candidates, list)
        or selected not in candidates
        or selected not in promoted
    ):
        raise LanguageControlCommandError("three-scene evidence did not lock one selected router")
    return (cast(str, selected),)


def _validate_prior_one_scene_authority(
    *,
    report_path: Path,
    verification_path: Path,
    schedule: ControlScheduleLock,
    corpus: GeneratedLanguageCorpus,
    registry: ControllerRegistry,
    source: SelectedNeuroSymbolicDispatchSource,
) -> dict[str, object]:
    """Rehash the completed one-scene parent and match its independent audit."""

    report = _read_object(
        _resolved_unlinked(report_path, label="prior one-scene stage report"),
        label="prior one-scene stage report",
    )
    verification = _read_object(
        _resolved_unlinked(verification_path, label="prior one-scene verification"),
        label="prior one-scene verification",
    )
    one_scene_inputs = prepare_development_inputs(
        schedule=schedule,
        corpus=corpus,
        stage=M5AStage.ONE_SCENE_CONTROL_SMOKE,
    )
    recomputed = verify_neuro_symbolic_control_evidence(
        report,
        inputs=one_scene_inputs,
        registry=registry,
        source=source,
    )
    required_equal = (
        "control_run_fingerprint",
        "control_record_set_fingerprint",
        "control_completion_fingerprint",
        "dispatch_source_fingerprint",
        "controller_registry_fingerprint",
    )
    if (
        verification.get("passed") is not True
        or verification.get("physical_target_validated") is not True
        or any(verification.get(key) != recomputed.get(key) for key in required_equal)
    ):
        raise LanguageControlCommandError(
            "prior one-scene report does not match its independent physical verification"
        )
    return {
        "report_path": str(report_path),
        "verification_path": str(verification_path),
        "run_fingerprint": recomputed["control_run_fingerprint"],
        "record_set_fingerprint": recomputed["control_record_set_fingerprint"],
        "completion_fingerprint": recomputed["control_completion_fingerprint"],
        "verification_schema": verification.get("schema_version"),
        "validated": True,
    }


def _validate_prior_three_scene_authority(
    *,
    report_path: Path,
    verification_path: Path,
    schedule: ControlScheduleLock,
    corpus: GeneratedLanguageCorpus,
    registry: ControllerRegistry,
    source: SelectedNeuroSymbolicDispatchSource,
) -> dict[str, object]:
    """Rehash the three-scene parent before authorizing the six-scene stage."""

    report = _read_object(
        _resolved_unlinked(report_path, label="prior three-scene stage report"),
        label="prior three-scene stage report",
    )
    verification = _read_object(
        _resolved_unlinked(verification_path, label="prior three-scene verification"),
        label="prior three-scene verification",
    )
    evidence_root = report.get("three_scene_evidence_root")
    prior_one = report.get("prior_one_scene_authority")
    verifier_git_commit = verification.get("verifier_git_commit")
    if (
        not isinstance(evidence_root, str)
        or not isinstance(prior_one, Mapping)
        or not isinstance(verifier_git_commit, str)
        or report.get("stage") != M5AStage.THREE_SCENE_CONTROL_SCREEN.value
        or report.get("passed") is not True
        or report.get("three_scene_control_screen_passed") is not True
        or report.get("full_control_development_authorized") is not True
        or report.get("full_control_development_completed") is not False
    ):
        raise LanguageControlCommandError("prior three-scene authority is incomplete")
    forbidden = (
        "language_final_accessed",
        "control_final_accessed",
        "test_split_accessed",
        "historical_fresh_accessed",
        "m42_final_accessed",
        "smolvla_go",
    )
    if any(report.get(key) is not False for key in forbidden):
        raise LanguageControlCommandError("prior three-scene stage accessed a prohibited source")
    three_inputs = prepare_development_inputs(
        schedule=schedule,
        corpus=corpus,
        stage=M5AStage.THREE_SCENE_CONTROL_SCREEN,
    )
    recomputed = verify_three_scene_control_evidence(
        evidence_root,
        inputs=three_inputs,
        registry=registry,
        source=source,
        prior_one_scene_authority=prior_one,
        expected_git_commit=verifier_git_commit,
    )
    required_true = (
        "passed",
        "physical_target_validated",
        "three_scene_control_screen_completed",
        "three_scene_control_screen_passed",
        "selected_router_locked",
        "full_control_development_authorized",
    )
    required_equal = (
        "run_fingerprint",
        "artifact_fingerprint",
        "completion_fingerprint",
        "control_run_fingerprint",
        "control_record_set_fingerprint",
    )
    if (
        any(verification.get(key) is not True for key in required_true)
        or any(verification.get(key) != recomputed.get(key) for key in required_equal)
        or any(verification.get(key) is not False for key in forbidden)
    ):
        raise LanguageControlCommandError(
            "prior three-scene report differs from independent physical verification"
        )
    return {
        "report_path": str(report_path),
        "verification_path": str(verification_path),
        "evidence_root": evidence_root,
        "run_fingerprint": recomputed["run_fingerprint"],
        "artifact_fingerprint": recomputed["artifact_fingerprint"],
        "completion_fingerprint": recomputed["completion_fingerprint"],
        "control_run_fingerprint": recomputed["control_run_fingerprint"],
        "control_record_set_fingerprint": recomputed["control_record_set_fingerprint"],
        "prior_one_scene_authority": dict(prior_one),
        "verification_schema": verification.get("schema_version"),
        "validated": True,
    }


def execute(args: argparse.Namespace) -> dict[str, object]:
    router_source_mode = _router_source_mode(args)
    output_root, _report_path = _safe_paths(args)
    corpus = build_language_corpus()
    schedule = load_authoritative_development_schedule(
        args.control_schedule,
        corpus=corpus,
    )
    stage = M5AStage(args.stage)
    inputs = prepare_development_inputs(schedule=schedule, corpus=corpus, stage=stage)
    budget = M5A_PHYSICAL_STAGE_BUDGETS[stage]
    effective_candidate_limit = (
        1
        if router_source_mode == NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL
        and stage in {M5AStage.THREE_SCENE_CONTROL_SCREEN, M5AStage.FULL_CONTROL_DEVELOPMENT}
        else budget.maximum_learned_candidates
    )
    base: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        "schedule_id": inputs.schedule.schedule_id,
        "schedule_fingerprint": inputs.schedule.schedule_fingerprint,
        "corpus_fingerprint": inputs.corpus_fingerprint,
        "episode_count_per_router": len(inputs.episodes),
        "stage": stage.value,
        "router_source_mode": router_source_mode,
        "router_order": ["oracle", "promoted_learned_only"],
        "maximum_learned_candidates": effective_candidate_limit,
        "maximum_expected_episode_atoms": len(inputs.episodes) * (1 + effective_candidate_limit),
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "smolvla_go": False,
        "physical_execution": False,
    }
    if args.dry_run:
        return {**base, "passed": True, "dry_run": True}

    git = inspect_git_state(PROJECT_ROOT)
    if not git.baseline_tracked or git.dirty:
        raise LanguageControlCommandError(
            "M5A target-development control requires a clean tracked Git worktree"
        )
    registry, locators = _load_frozen_controller_registry(
        checkpoint_root=args.checkpoint_root,
        dataset_root=args.dataset_root,
        runtime_selection_path=args.runtime_selection,
    )
    rule = RuleRouterV0()
    classifier: FactorizedTextRouterV0 | None = None
    classifier_identity: dict[str, object] | None = None
    llm_config: StructuredLLMRouterConfig | None = None
    llm: StructuredLocalLLMRouterV0 | None = None
    router_evaluation_identity: dict[str, object] | None = None
    neuro_symbolic_source = None
    neuro_symbolic_binding = None
    neuro_symbolic_router = None
    prior_one_scene_authority: dict[str, object] | None = None
    prior_three_scene_authority: dict[str, object] | None = None
    sealed_final_authority: dict[str, object] | None = None
    if router_source_mode == NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL:
        assert args.neuro_symbolic_evidence_root is not None
        neuro_symbolic_source = validate_selected_neuro_symbolic_dispatch_source(
            args.neuro_symbolic_evidence_root,
            corpus=corpus,
        )
        neuro_symbolic_binding = bind_selected_router_to_controller_registry(
            neuro_symbolic_source,
            registry,
        )
        if stage is M5AStage.THREE_SCENE_CONTROL_SCREEN:
            assert args.prior_stage_report is not None
            assert args.prior_stage_verification is not None
            prior_one_scene_authority = _validate_prior_one_scene_authority(
                report_path=args.prior_stage_report,
                verification_path=args.prior_stage_verification,
                schedule=schedule,
                corpus=corpus,
                registry=registry,
                source=neuro_symbolic_source,
            )
        elif stage is M5AStage.FULL_CONTROL_DEVELOPMENT:
            assert args.prior_stage_report is not None
            assert args.prior_stage_verification is not None
            prior_three_scene_authority = _validate_prior_three_scene_authority(
                report_path=args.prior_stage_report,
                verification_path=args.prior_stage_verification,
                schedule=schedule,
                corpus=corpus,
                registry=registry,
                source=neuro_symbolic_source,
            )
            sealed_final_authority = load_sealed_final_authority(
                args.control_schedule,
                corpus=corpus,
            )
        neuro_symbolic_router = load_selected_neuro_symbolic_router(
            neuro_symbolic_source,
            corpus=corpus,
            device=args.device,
            local_files_only=True,
        )
        candidates = (NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL,)
    else:
        assert args.classifier_artifact_root is not None
        assert args.router_evaluation_evidence_root is not None
        assert args.llm_model_id is not None
        assert args.llm_model_revision is not None
        assert args.llm_tokenizer_revision is not None
        assert args.llm_license is not None
        classifier, classifier_identity = _classifier_router(
            args.classifier_artifact_root.resolve(), device=args.device
        )
        llm_config = StructuredLLMRouterConfig(
            model_id=args.llm_model_id,
            model_revision=args.llm_model_revision,
            tokenizer_revision=args.llm_tokenizer_revision,
            dtype=args.llm_dtype,
            maximum_new_tokens=args.llm_maximum_new_tokens,
        )
        prompt_examples, router_evaluation_identity = _load_frozen_router_evaluation(
            args.router_evaluation_evidence_root,
            corpus=corpus,
            classifier_identity=classifier_identity,
            rule=rule,
            llm_config=llm_config,
            llm_license=args.llm_license,
            llm_license_reviewed=bool(args.llm_license_reviewed),
        )
        llm_generator = TransformersLocalTextGenerator(
            config=llm_config,
            device=args.device,
            local_files_only=not args.allow_model_download,
        )
        llm = StructuredLocalLLMRouterV0(
            config=llm_config,
            generator=llm_generator,
            prompt_examples=prompt_examples,
        )
        candidates = _stage_candidates(
            stage=stage,
            router_evaluation_identity=router_evaluation_identity,
            prior_stage_report=args.prior_stage_report,
        )
    if not candidates:
        raise LanguageControlCommandError("no learned router was promoted to this physical stage")
    environment = _create_environment()
    try:
        loader = StrictPerTaskControllerLoader(
            registry=registry,
            locators=locators,
            schedule_digest=schedule.schedule_fingerprint,
        )
        rejection_noop_probe = run_rejection_noop_probe(
            registry=registry,
            loader=loader,
            environment=environment,
            router=(neuro_symbolic_router if neuro_symbolic_router is not None else rule),
        )
        rejection_probe_set = (
            run_three_scene_rejection_probes(
                registry=registry,
                loader=loader,
                environment=environment,
                router=neuro_symbolic_router,
            )
            if stage is M5AStage.THREE_SCENE_CONTROL_SCREEN and neuro_symbolic_router is not None
            else run_full_control_rejection_probes(
                registry=registry,
                loader=loader,
                environment=environment,
                router=neuro_symbolic_router,
            )
            if stage is M5AStage.FULL_CONTROL_DEVELOPMENT and neuro_symbolic_router is not None
            else None
        )
        dispatcher = ControllerDispatcher(registry=registry, loader=loader)
        available: dict[str, DecisionProvider] = {}
        if classifier is not None and llm is not None:
            available.update(
                {
                    "classifier": _provider(classifier),
                    "llm": _provider(llm),
                }
            )
        if neuro_symbolic_router is not None:
            available[NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL] = _provider(neuro_symbolic_router)
        providers: dict[str, DecisionProvider] = {
            "oracle": _oracle_provider,
            **{candidate: available[candidate] for candidate in candidates},
        }
        result = run_development_control(
            inputs=inputs,
            registry=registry,
            dispatcher=dispatcher,
            environment=environment,
            providers=providers,
            output_root=output_root,
            run_identity={
                "git_commit": git.commit,
                "classifier": classifier_identity,
                "llm_config": None if llm_config is None else llm_config.to_dict(),
                "llm_config_fingerprint": (None if llm_config is None else llm_config.fingerprint),
                "llm_prompt_fingerprint": None if llm is None else llm.prompt_fingerprint,
                "llm_prompt_example_ids": (None if llm is None else list(llm.prompt_example_ids)),
                "router_evaluation": router_evaluation_identity,
                "neuro_symbolic_dispatch_source": (
                    None if neuro_symbolic_source is None else neuro_symbolic_source.identity_dict()
                ),
                "neuro_symbolic_controller_binding": (
                    None if neuro_symbolic_binding is None else neuro_symbolic_binding.to_dict()
                ),
                "rule_config_fingerprint": rule.config.fingerprint,
                "candidate_promotion_identity": {
                    "promoted_learned_routers": list(candidates),
                    "primary_learned_router": candidates[0],
                    "stage_candidates": list(candidates),
                    "prior_stage_report": (
                        None if args.prior_stage_report is None else str(args.prior_stage_report)
                    ),
                    "prior_one_scene_authority": prior_one_scene_authority,
                    "prior_three_scene_authority": prior_three_scene_authority,
                },
                "runtime_selection_fingerprint": (
                    registry.entries[0].runtime_selection_fingerprint
                ),
            },
            require_active_episode_spec=True,
            rejection_noop_probe=rejection_noop_probe,
        )
        if stage is M5AStage.THREE_SCENE_CONTROL_SCREEN:
            if (
                rejection_probe_set is None
                or prior_one_scene_authority is None
                or neuro_symbolic_source is None
                or neuro_symbolic_binding is None
            ):
                raise LanguageControlCommandError("three-scene screen lacks immutable authority")
            result = {
                **result,
                **_finalize_three_scene_screen(
                    output_root=output_root,
                    inputs=inputs,
                    registry=registry,
                    control_result=result,
                    rejection_probe_set=rejection_probe_set,
                    prior_authority=prior_one_scene_authority,
                    dispatch_source=neuro_symbolic_source.to_dict(),
                    controller_binding=neuro_symbolic_binding.to_dict(),
                    git_commit=git.commit,
                ),
            }
        elif stage is M5AStage.FULL_CONTROL_DEVELOPMENT:
            if (
                rejection_probe_set is None
                or prior_three_scene_authority is None
                or sealed_final_authority is None
                or neuro_symbolic_source is None
                or neuro_symbolic_binding is None
            ):
                raise LanguageControlCommandError("full development lacks immutable authority")
            result = {
                **result,
                **_finalize_full_control_development(
                    output_root=output_root,
                    inputs=inputs,
                    registry=registry,
                    control_result=result,
                    rejection_probe_set=rejection_probe_set,
                    prior_authority=prior_three_scene_authority,
                    sealed_final_authority=sealed_final_authority,
                    dispatch_source=neuro_symbolic_source.to_dict(),
                    controller_binding=neuro_symbolic_binding.to_dict(),
                    git_commit=git.commit,
                ),
            }
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    return {
        **base,
        **result,
        "controller_registry": registry.to_dict(),
        "classifier_identity": classifier_identity,
        "llm_config": None if llm_config is None else llm_config.to_dict(),
        "llm_prompt_fingerprint": None if llm is None else llm.prompt_fingerprint,
        "router_evaluation_identity": router_evaluation_identity,
        "neuro_symbolic_dispatch_source": (
            None if neuro_symbolic_source is None else neuro_symbolic_source.to_dict()
        ),
        "neuro_symbolic_controller_binding": (
            None if neuro_symbolic_binding is None else neuro_symbolic_binding.to_dict()
        ),
        "rejection_noop_probe": rejection_noop_probe,
        "prior_one_scene_authority": prior_one_scene_authority,
        "prior_three_scene_authority": prior_three_scene_authority,
        "sealed_final_authority": sealed_final_authority,
        "physical_execution": True,
        "dry_run": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path: Path | None = None
    try:
        _output_root, report_path = _safe_output_paths(args.output_root, args.report)
        report = execute(args)
    except Exception as error:  # noqa: BLE001 - command boundary preserves exact failure
        traceback.print_exc()
        report = {
            "schema_version": COMMAND_SCHEMA,
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
            "language_final_accessed": False,
            "control_final_accessed": False,
            "m42_final_accessed": False,
            "test_split_accessed": False,
            "historical_fresh_accessed": False,
            "smolvla_go": False,
            "physical_execution": False,
        }
    if report_path is not None:
        try:
            _write_report(report_path, report)
        except Exception:
            traceback.print_exc()
            return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
