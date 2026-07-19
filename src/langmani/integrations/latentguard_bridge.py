"""Strict JSON bridge for the bounded LatentGuard M6B initial-state smoke.

The bridge is deliberately action-only and initial-state-only.  It can reset one
environment and query one accepted PerTask ACT checkpoint, but it never calls
``env.step`` and never creates simulator outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import torch

from langmani.datasets.identity import canonical_json
from langmani.environments.specs import canonical_instruction, stable_task_id
from langmani.language.controller_registry import (
    LOCKED_CONTROL_MODE,
    LOCKED_ENVIRONMENT_ID,
    LOCKED_EXECUTION_HORIZON,
    ControllerRegistry,
    ControllerRegistryEntry,
    ControllerRegistryLocators,
    load_controller_registry_metadata,
    validate_active_controller_environment,
)
from langmani.policies.act_action_bounds import BoundedActionEnvPostprocessorV0
from langmani.policies.act_rollout import build_policy_observation
from langmani.policies.act_types import ACT_ACTION_COMPONENTS, ActVariant
from langmani.policies.m42_evaluation import M42PolicyKind, load_m4_checkpoint_context

POLICY_BINDING_SCHEMA = "LangManiLatentGuardPolicyBindingV1"
INITIAL_PROPOSAL_SCHEMA = "LangManiLatentGuardInitialProposalV1"
CANDIDATE_REQUEST_SCHEMA = "LatentGuardRawCandidateRequestV1"
PROJECTED_CANDIDATES_SCHEMA = "LangManiProjectedCandidatesV1"
BRIDGE_AUDIT_SCHEMA = "LangManiLatentGuardBridgeAuditV1"
PROJECTION_SEMANTIC = "langmani_existing_active_action_bounds_projection_v1"
INITIAL_RESET_SEMANTIC = "seeded_initial_reset_only_no_policy_state_restore_v1"
PROBE_SEED_SEMANTIC = "sha256_prefix_u31_m6a1_bridge_probe_v1"
EXPECTED_POLICY_CHUNK = 50
EXPECTED_PREFIX_HORIZON = 10
EXPECTED_CANDIDATE_COUNT = 4
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class LatentGuardBridgeError(RuntimeError):
    """Raised when a bridge identity, artifact, or zero-action gate differs."""


def _fail(context: str, reason: str) -> NoReturn:
    raise LatentGuardBridgeError(f"{context}: {reason}")


def _canonical_bytes(value: object) -> bytes:
    try:
        return canonical_json(value).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LatentGuardBridgeError(f"canonical JSON failed: {error}") from error


def content_digest(value: object) -> str:
    """Return the canonical SHA-256 digest for one finite JSON value."""

    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _exact_mapping(value: object, fields: set[str], context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected an object with text keys")
    result = cast(Mapping[str, object], value)
    if set(result) != fields:
        _fail(context, "unexpected or missing fields")
    return result


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _digest(value: object, context: str) -> str:
    result = _text(value, context)
    if _DIGEST_RE.fullmatch(result) is None:
        _fail(context, "expected a prefixed SHA-256 digest")
    return result


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(context, f"expected integer >= {minimum}")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        _fail(context, "expected a boolean")
    return value


def _finite_array(value: object, *, shape: tuple[int, ...], context: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise LatentGuardBridgeError(f"{context}: invalid numeric array") from error
    if array.shape != shape or not bool(np.all(np.isfinite(array))):
        _fail(context, f"expected finite float32{shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def make_envelope(schema_version: str, payload: Mapping[str, object]) -> dict[str, object]:
    """Build a strict self-digesting JSON envelope."""

    schema = _text(schema_version, "schema_version")
    body = json.loads(_canonical_bytes(dict(payload)).decode("utf-8"))
    return {
        "content_digest": content_digest({"payload": body, "schema_version": schema}),
        "payload": body,
        "schema_version": schema,
    }


def validate_envelope(value: object, *, expected_schema: str) -> Mapping[str, object]:
    """Validate exact envelope fields, schema, finite JSON, and content digest."""

    item = _exact_mapping(value, {"content_digest", "payload", "schema_version"}, "bridge envelope")
    if _text(item["schema_version"], "bridge envelope.schema_version") != expected_schema:
        _fail("bridge envelope.schema_version", "unexpected schema")
    payload = item["payload"]
    if not isinstance(payload, Mapping):
        _fail("bridge envelope.payload", "expected an object")
    expected = content_digest({"payload": dict(payload), "schema_version": expected_schema})
    if _digest(item["content_digest"], "bridge envelope.content_digest") != expected:
        _fail("bridge envelope.content_digest", "content changed")
    return cast(Mapping[str, object], payload)


def read_envelope(
    path: Path, *, expected_schema: str
) -> tuple[dict[str, object], Mapping[str, object]]:
    """Read a UTF-8 strict JSON envelope and reject duplicate keys and constants."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                _fail("JSON", f"duplicate field {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> NoReturn:
        _fail("JSON", f"non-finite constant {value!r}")

    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LatentGuardBridgeError(f"cannot read bridge JSON: {error}") from error
    if not isinstance(raw, dict):
        _fail("bridge JSON", "expected one object")
    payload = validate_envelope(raw, expected_schema=expected_schema)
    return raw, payload


def write_envelope(path: Path, envelope: Mapping[str, object]) -> None:
    """Write canonical UTF-8 JSON without non-semantic runtime metadata."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical_bytes(dict(envelope)) + b"\n")


def _git_identity(root: Path, *, require_clean: bool) -> tuple[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            _fail("Git", result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    sha = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--short")
    if _SHA_RE.fullmatch(sha) is None:
        _fail("Git", "HEAD is not a full SHA")
    if require_clean and status:
        _fail("Git", "proposal export requires a clean tracked checkout")
    return sha, branch


def derive_probe_seed() -> int:
    """Return the reserved bridge-only seed from a checked-in semantic string."""

    digest = hashlib.sha256(PROBE_SEED_SEMANTIC.encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _load_registry(path: Path) -> ControllerRegistry:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LatentGuardBridgeError(f"cannot read controller registry: {error}") from error
    if not isinstance(raw, Mapping):
        _fail("controller registry", "expected an object")
    return ControllerRegistry.from_dict(raw)


def _select_entry(registry: ControllerRegistry) -> ControllerRegistryEntry:
    eligible = sorted(
        entry.task_id
        for entry in registry.entries
        if stable_task_id(entry.task_spec) == entry.task_id
    )
    if not eligible:
        _fail("controller selection", "no canonical digest-valid controller is eligible")
    selected = eligible[0]
    matches = tuple(entry for entry in registry.entries if entry.task_id == selected)
    if len(matches) != 1:
        _fail("controller selection", "selected task lacks one unique controller")
    return matches[0]


def export_policy_binding(
    *,
    registry_path: Path,
    checkpoint_root: Path,
    dataset_root: Path,
    runtime_selection_path: Path,
    repository_root: Path,
) -> dict[str, object]:
    """Select and bind one accepted PerTask controller without outcome access."""

    recorded = _load_registry(registry_path)
    rebuilt, locators = load_controller_registry_metadata(
        checkpoint_root=checkpoint_root,
        dataset_root=dataset_root,
        runtime_selection_path=runtime_selection_path,
    )
    if recorded != rebuilt or recorded.registry_fingerprint != locators.registry_fingerprint:
        _fail("controller registry", "recorded and artifact-rebuilt registries differ")
    entry = _select_entry(recorded)
    location = locators.require(entry.task_id)
    if not location.run_root.is_dir():
        _fail("controller selection", "selected controller runtime is absent")
    bridge_sha, bridge_branch = _git_identity(repository_root, require_clean=True)
    return build_policy_binding(
        registry=recorded,
        entry=entry,
        bridge_git_commit=bridge_sha,
        bridge_git_branch=bridge_branch,
    )


def build_policy_binding(
    *,
    registry: ControllerRegistry,
    entry: ControllerRegistryEntry,
    bridge_git_commit: str,
    bridge_git_branch: str,
) -> dict[str, object]:
    """Build a path-free policy binding from a validated registry entry."""

    canonical = registry.require(entry.task_id)
    if canonical != entry or _select_entry(registry) != entry:
        _fail("controller selection", "entry is not the deterministic selected controller")
    if _SHA_RE.fullmatch(bridge_git_commit) is None:
        _fail("bridge_git_commit", "expected a full Git SHA")
    _text(bridge_git_branch, "bridge_git_branch")
    action_contract = dict(entry.action_space_contract)
    payload: dict[str, object] = {
        "action_bound_config": dict(entry.action_bound_config.to_dict()),
        "action_bounds_digest": content_digest(action_contract),
        "action_space_contract": action_contract,
        "action_dimension": ACT_ACTION_COMPONENTS,
        "action_shape": [ACT_ACTION_COMPONENTS],
        "act_chunk_length": int(cast(int, entry.model_config["chunk_size"])),
        "bridge_git_branch": bridge_git_branch,
        "bridge_git_commit": bridge_git_commit,
        "checkpoint_digest": entry.checkpoint_fingerprint,
        "checkpoint_step": entry.checkpoint_step,
        "control_frequency_hz": int(cast(int, entry.rollout_config["control_frequency_hz"])),
        "control_mode": LOCKED_CONTROL_MODE,
        "controller_registry_digest": registry.registry_fingerprint,
        "destination_identity": entry.task_spec.target_bin_id,
        "environment_id": LOCKED_ENVIRONMENT_ID,
        "execution_queue_horizon": LOCKED_EXECUTION_HORIZON,
        "instruction": canonical_instruction(entry.task_spec),
        "model_configuration_digest": entry.model_config_fingerprint,
        "normalization_identity": entry.train_statistics_fingerprint,
        "object_identity": entry.task_spec.target_object_id,
        "postprocessor_digest": entry.postprocessor_fingerprint,
        "preprocessor_digest": entry.preprocessor_fingerprint,
        "producer_langmani_sha": entry.producer_git_commit,
        "selection_semantic": "lexically_first_eligible_canonical_task_v1",
        "task_id": entry.task_id,
        "task_semantic": dict(entry.task_spec.to_dict()),
    }
    return make_envelope(POLICY_BINDING_SCHEMA, payload)


class _CountingEnvironment:
    """Transparent environment proxy proving the accepted probe never steps."""

    def __init__(self, environment: object) -> None:
        self.environment = environment
        self.reset_count = 0
        self.step_count = 0

    def reset(self, *args: object, **kwargs: object) -> object:
        self.reset_count += 1
        return cast(Any, self.environment).reset(*args, **kwargs)

    def step(self, *args: object, **kwargs: object) -> object:
        self.step_count += 1
        _fail("zero-action probe", "env.step is prohibited")

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> object:
        return getattr(self.environment, name)


def _create_environment() -> object:
    import gymnasium as gym

    import langmani.environments  # noqa: F401
    from langmani.environments.pick_place_by_instruction import ENV_ID

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode=LOCKED_CONTROL_MODE,
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )


def _retarget_context(context: object, *, device: str) -> None:
    loaded = getattr(context, "loaded", None)
    policy = getattr(loaded, "policy", None)
    target = torch.device(device)
    move = getattr(policy, "to", None)
    evaluate = getattr(policy, "eval", None)
    if not callable(move) or not callable(evaluate):
        _fail("checkpoint", "loaded policy lacks to()/eval()")
    config = getattr(policy, "config", None)
    if config is not None and hasattr(config, "device"):
        config.device = device
    move(target)
    evaluate()
    for pipeline in (
        getattr(loaded, "preprocessor", None),
        getattr(loaded, "postprocessor", None),
    ):
        steps = getattr(pipeline, "steps", None)
        if not isinstance(steps, Sequence):
            _fail("checkpoint", "processor lacks public steps")
        for step in steps:
            if hasattr(step, "device"):
                step.device = target
            if hasattr(step, "stats") and hasattr(step, "_tensor_stats"):
                initialize = getattr(step, "__post_init__", None)
                if not callable(initialize):
                    _fail("checkpoint", "normalization processor cannot retarget")
                initialize()


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return content_digest(
        {
            "bytes_sha256": f"sha256:{hashlib.sha256(array.tobytes(order='C')).hexdigest()}",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    )


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return content_digest(
        {
            "bytes_sha256": f"sha256:{hashlib.sha256(array.tobytes(order='C')).hexdigest()}",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    )


def _candidate_pool_digest(
    *,
    candidate_source_digest: str,
    policy_binding_digest: str,
    proposal_digest: str,
    raw_candidate_digests: Sequence[str],
) -> str:
    return content_digest(
        {
            "candidate_source_digest": candidate_source_digest,
            "policy_binding_digest": policy_binding_digest,
            "proposal_digest": proposal_digest,
            "raw_candidate_digests": list(raw_candidate_digests),
        }
    )


def _opaque_candidate_id(*, candidate_pool_digest: str, ordinal: int) -> str:
    return (
        "lgc-sha256-"
        + hashlib.sha256(
            _canonical_bytes({"candidate_pool_digest": candidate_pool_digest, "ordinal": ordinal})
        ).hexdigest()
    )


def _binding_payload(envelope: object) -> Mapping[str, object]:
    payload = validate_envelope(envelope, expected_schema=POLICY_BINDING_SCHEMA)
    required = {
        "action_bound_config",
        "action_bounds_digest",
        "action_space_contract",
        "action_dimension",
        "action_shape",
        "act_chunk_length",
        "bridge_git_branch",
        "bridge_git_commit",
        "checkpoint_digest",
        "checkpoint_step",
        "control_frequency_hz",
        "control_mode",
        "controller_registry_digest",
        "destination_identity",
        "environment_id",
        "execution_queue_horizon",
        "instruction",
        "model_configuration_digest",
        "normalization_identity",
        "object_identity",
        "postprocessor_digest",
        "preprocessor_digest",
        "producer_langmani_sha",
        "selection_semantic",
        "task_id",
        "task_semantic",
    }
    if set(payload) != required:
        _fail("policy binding", "unexpected or missing fields")
    if (
        _integer(payload["action_dimension"], "action_dimension", minimum=1)
        != ACT_ACTION_COMPONENTS
        or _integer(payload["act_chunk_length"], "act_chunk_length", minimum=1)
        != EXPECTED_POLICY_CHUNK
        or _integer(payload["execution_queue_horizon"], "execution_queue_horizon", minimum=1)
        != EXPECTED_PREFIX_HORIZON
    ):
        _fail("policy binding", "action dimensions or horizons differ")
    return payload


def _validate_binding_against_registry(
    binding: Mapping[str, object], registry: ControllerRegistry
) -> ControllerRegistryEntry:
    if binding["controller_registry_digest"] != registry.registry_fingerprint:
        _fail("policy binding", "controller registry differs")
    entry = registry.require(_text(binding["task_id"], "policy binding.task_id"))
    checks = {
        "checkpoint_digest": entry.checkpoint_fingerprint,
        "model_configuration_digest": entry.model_config_fingerprint,
        "normalization_identity": entry.train_statistics_fingerprint,
        "postprocessor_digest": entry.postprocessor_fingerprint,
        "preprocessor_digest": entry.preprocessor_fingerprint,
        "producer_langmani_sha": entry.producer_git_commit,
    }
    if any(binding[name] != expected for name, expected in checks.items()):
        _fail("policy binding", "controller or processor identity differs")
    return entry


def export_initial_proposal(
    *,
    binding_envelope: Mapping[str, object],
    registry: ControllerRegistry,
    locators: ControllerRegistryLocators,
    repository_root: Path,
    probe_seed: int,
    device: str,
    environment_factory: Callable[[], object] = _create_environment,
    context_loader: Callable[..., object] = load_m4_checkpoint_context,
) -> dict[str, object]:
    """Reset once, query once, project once, and execute zero actions."""

    if probe_seed != derive_probe_seed():
        _fail("probe seed", "must equal the checked-in bridge-only seed")
    if device not in {"cpu", "cuda"}:
        _fail("device", "expected cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        _fail("device", "CUDA is unavailable")
    binding = _binding_payload(binding_envelope)
    current_sha, _ = _git_identity(repository_root, require_clean=True)
    if binding["bridge_git_commit"] != current_sha:
        _fail("policy binding", "bridge Git SHA differs from current checkout")
    entry = _validate_binding_against_registry(binding, registry)
    if locators.registry_fingerprint != registry.registry_fingerprint:
        _fail("controller locators", "registry identity differs")
    location = locators.require(entry.task_id)
    environment = _CountingEnvironment(environment_factory())
    policy_query_count = 0
    try:
        validate_active_controller_environment(environment, registry)
        context = context_loader(
            location.run_root,
            policy_kind=M42PolicyKind.PER_TASK,
            expected_checkpoint_fingerprint=entry.checkpoint_fingerprint,
            expected_run_fingerprint=entry.run_fingerprint,
            expected_dataset_fingerprint=entry.m3b_export_fingerprint,
            expected_task_id=entry.task_id,
        )
        _retarget_context(context, device=device)
        loaded = cast(Any, context).loaded
        for component, name in (
            (loaded.policy, "policy"),
            (loaded.preprocessor, "preprocessor"),
            (loaded.postprocessor, "postprocessor"),
        ):
            reset = getattr(component, "reset", None)
            if not callable(reset):
                _fail("checkpoint", f"{name} lacks reset()")
            reset()
        begin_task = getattr(loaded.policy, "begin_task", None)
        if callable(begin_task):
            begin_task(entry.task_id)
        reset_result = environment.reset(
            seed=probe_seed, options={"task_spec": entry.task_spec.to_dict()}
        )
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            _fail("environment reset", "expected Gymnasium reset two-tuple")
        observation, _ = reset_result
        base = getattr(environment, "unwrapped", environment)
        raw_observation = build_policy_observation(
            observation,
            base_environment=base,
            variant=ActVariant.PER_TASK,
            task_id=entry.task_id,
            task_conditioner=None,
        )
        processed = loaded.preprocessor(raw_observation)
        if not isinstance(processed, Mapping):
            _fail("policy query", "preprocessor returned a non-mapping")
        with torch.inference_mode():
            predicted = loaded.policy.predict_action_chunk(dict(processed))
            policy_query_count += 1
            postprocessed = loaded.postprocessor(predicted)
        if not isinstance(postprocessed, torch.Tensor):
            _fail("policy query", "postprocessor returned a non-tensor")
        raw = postprocessed.detach().cpu().numpy()
        if raw.shape != (1, EXPECTED_POLICY_CHUNK, ACT_ACTION_COMPONENTS):
            _fail("policy query", "postprocessed chunk must be float[1,50,8]")
        raw_chunk = np.ascontiguousarray(raw[0], dtype=np.float32)
        if not bool(np.all(np.isfinite(raw_chunk))):
            _fail("policy query", "postprocessed chunk contains non-finite values")
        projector = BoundedActionEnvPostprocessorV0.from_environment(
            environment,
            entry.action_bound_config,
            expected_action_components=ACT_ACTION_COMPONENTS,
        )
        action_contract = projector.action_space_contract()
        if content_digest(action_contract) != binding["action_bounds_digest"]:
            _fail("action bounds", "active environment bounds differ from binding")
        projected_result = projector.process(raw_chunk, rollout_step=1)
        projected_chunk = np.ascontiguousarray(
            cast(np.ndarray, projected_result.executed_action), dtype=np.float32
        )
        record = projected_result.audit_record
        violation_mask = np.asarray(record.violation_mask, dtype=np.bool_)
        if violation_mask.shape != raw_chunk.shape:
            _fail("projection", "correction mask shape differs")
        image = raw_observation.get("observation.images.base_camera")
        state = raw_observation.get("observation.state")
        if not isinstance(image, torch.Tensor) or not isinstance(state, torch.Tensor):
            _fail("policy observation", "canonical tensors are absent")
        if environment.reset_count != 1 or environment.step_count != 0:
            _fail("zero-action probe", "expected exactly one reset and zero steps")
        payload: dict[str, object] = {
            "action_bounds": action_contract,
            "action_bounds_digest": binding["action_bounds_digest"],
            "environment_step_count": environment.step_count,
            "initial_reset_semantic": INITIAL_RESET_SEMANTIC,
            "policy_binding_digest": cast(str, binding_envelope["content_digest"]),
            "policy_observation_identity": {
                "image_digest": _tensor_digest(image),
                "image_shape": list(image.shape),
                "raw_rgb_embedded": False,
                "state_digest": _tensor_digest(state),
                "state_shape": list(state.shape),
            },
            "policy_query_count": policy_query_count,
            "probe_seed": probe_seed,
            "probe_seed_semantic": PROBE_SEED_SEMANTIC,
            "projected_chunk": projected_chunk.tolist(),
            "projected_chunk_digest": _array_digest(projected_chunk),
            "projected_prefix": projected_chunk[:EXPECTED_PREFIX_HORIZON].tolist(),
            "projected_prefix_digest": _array_digest(projected_chunk[:EXPECTED_PREFIX_HORIZON]),
            "projection_evidence": {
                "correction_count": int(record.projected_component_count),
                "correction_mask": violation_mask.tolist(),
                "nonfinite_count": 0,
                "projection_semantic": PROJECTION_SEMANTIC,
            },
            "raw_postprocessed_chunk": raw_chunk.tolist(),
            "raw_postprocessed_chunk_digest": _array_digest(raw_chunk),
            "raw_prefix_digest": _array_digest(raw_chunk[:EXPECTED_PREFIX_HORIZON]),
            "reset_count": environment.reset_count,
            "task_id": entry.task_id,
            "zero_action_execution": True,
            "outcomes_generated": False,
        }
        return make_envelope(INITIAL_PROPOSAL_SCHEMA, payload)
    finally:
        environment.close()


def project_candidates(
    *,
    binding_envelope: Mapping[str, object],
    candidate_envelope: Mapping[str, object],
) -> dict[str, object]:
    """Project four raw prefixes with the existing LangMani bound processor."""

    binding = _binding_payload(binding_envelope)
    request = validate_envelope(candidate_envelope, expected_schema=CANDIDATE_REQUEST_SCHEMA)
    required = {
        "action_dimension",
        "candidate_count",
        "candidate_pool_digest",
        "candidate_source_digest",
        "candidates",
        "environment_actions_executed",
        "outcomes_available",
        "policy_binding_digest",
        "proposal_digest",
        "source_horizon",
    }
    if set(request) != required:
        _fail("candidate request", "unexpected or missing fields")
    if request["policy_binding_digest"] != binding_envelope["content_digest"]:
        _fail("candidate request", "policy binding differs")
    candidate_source_digest = _digest(
        request["candidate_source_digest"], "candidate request.candidate_source_digest"
    )
    candidate_pool_digest = _digest(
        request["candidate_pool_digest"], "candidate request.candidate_pool_digest"
    )
    proposal_digest = _digest(request["proposal_digest"], "candidate request.proposal_digest")
    if (
        _integer(request["action_dimension"], "action_dimension", minimum=1)
        != ACT_ACTION_COMPONENTS
        or _integer(request["source_horizon"], "source_horizon", minimum=1)
        != EXPECTED_PREFIX_HORIZON
        or _integer(request["candidate_count"], "candidate_count", minimum=1)
        != EXPECTED_CANDIDATE_COUNT
    ):
        _fail("candidate request", "count or shape contract differs")
    if _boolean(request["environment_actions_executed"], "environment_actions_executed"):
        _fail("candidate request", "executed actions are prohibited")
    if _boolean(request["outcomes_available"], "outcomes_available"):
        _fail("candidate request", "outcomes are prohibited")
    candidates = request["candidates"]
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        _fail("candidate request", "candidates must be an array")
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        _fail("candidate request", "expected exactly four candidates")
    contract = binding["action_bound_config"]
    if not isinstance(contract, Mapping):
        _fail("policy binding.action_bound_config", "expected an object")
    from langmani.policies.act_action_bounds import ActionBoundConfig

    raw_entries: list[Mapping[str, object]] = []
    for index, item in enumerate(candidates):
        raw_entries.append(
            _exact_mapping(
                item,
                {
                    "candidate_id",
                    "raw_actions",
                    "raw_candidate_digest",
                    "transformation_config_digest",
                    "transformation_id",
                },
                f"candidate[{index}]",
            )
        )
    # The bound values travel in the signed policy binding.  This permits the
    # existing projector to run without a second environment or reset.
    action_space_contract = binding.get("action_space_contract")
    if not isinstance(action_space_contract, Mapping):
        _fail("policy binding", "action_space_contract is required for projection")
    if content_digest(dict(action_space_contract)) != binding["action_bounds_digest"]:
        _fail("policy binding", "action-space contract digest differs")
    low = action_space_contract.get("lower_bounds")
    high = action_space_contract.get("upper_bounds")
    projector = BoundedActionEnvPostprocessorV0(
        ActionBoundConfig.from_dict(contract), low=low, high=high
    )
    projected: list[dict[str, object]] = []
    seen: set[str] = set()
    raw_digests: list[str] = []
    for index, item in enumerate(raw_entries):
        candidate_id = _text(item["candidate_id"], f"candidate[{index}].candidate_id")
        if candidate_id in seen:
            _fail("candidate request", "candidate IDs must be unique")
        seen.add(candidate_id)
        raw = _finite_array(
            item["raw_actions"],
            shape=(EXPECTED_PREFIX_HORIZON, ACT_ACTION_COMPONENTS),
            context=f"candidate[{index}].raw_actions",
        )
        raw_digest = _digest(
            item["raw_candidate_digest"], f"candidate[{index}].raw_candidate_digest"
        )
        if _array_digest(raw) != raw_digest:
            _fail(f"candidate[{index}]", "raw candidate digest differs")
        raw_digests.append(raw_digest)
        _digest(
            item["transformation_config_digest"],
            f"candidate[{index}].transformation_config_digest",
        )
        _text(item["transformation_id"], f"candidate[{index}].transformation_id")
        result = projector.process(raw, rollout_step=1)
        values = np.ascontiguousarray(cast(np.ndarray, result.executed_action), dtype=np.float32)
        mask = np.asarray(result.audit_record.violation_mask, dtype=np.bool_)
        projected.append(
            {
                "candidate_id": candidate_id,
                "correction_count": int(result.audit_record.projected_component_count),
                "correction_mask": mask.tolist(),
                "nonfinite_count": 0,
                "projected_actions": values.tolist(),
                "projected_candidate_digest": _array_digest(values),
                "projection_semantic": PROJECTION_SEMANTIC,
                "raw_candidate_digest": item["raw_candidate_digest"],
                "transformation_config_digest": item["transformation_config_digest"],
                "transformation_id": item["transformation_id"],
            }
        )
    expected_pool_digest = _candidate_pool_digest(
        candidate_source_digest=candidate_source_digest,
        policy_binding_digest=cast(str, binding_envelope["content_digest"]),
        proposal_digest=proposal_digest,
        raw_candidate_digests=raw_digests,
    )
    if candidate_pool_digest != expected_pool_digest:
        _fail("candidate request", "candidate pool digest differs")
    for ordinal, item in enumerate(projected):
        if item["candidate_id"] != _opaque_candidate_id(
            candidate_pool_digest=candidate_pool_digest, ordinal=ordinal
        ):
            _fail("candidate request", "opaque candidate ID differs")
    payload = {
        "action_bounds_digest": binding["action_bounds_digest"],
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "candidate_pool_digest": candidate_pool_digest,
        "candidates": projected,
        "environment_step_count": 0,
        "outcomes_available": False,
        "policy_binding_digest": binding_envelope["content_digest"],
        "projection_semantic": PROJECTION_SEMANTIC,
        "raw_candidate_manifest_digest": candidate_envelope["content_digest"],
        "task_id": binding["task_id"],
        "zero_action_execution": True,
    }
    return make_envelope(PROJECTED_CANDIDATES_SCHEMA, payload)


def audit_bridge(
    *,
    binding_envelope: Mapping[str, object],
    proposal_envelope: Mapping[str, object] | None = None,
    candidate_envelope: Mapping[str, object] | None = None,
    projected_envelope: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Audit available bridge stages without importing LatentGuard."""

    binding = _binding_payload(binding_envelope)
    gates: dict[str, bool] = {"policy_binding_valid": bool(binding)}
    if proposal_envelope is not None:
        proposal = validate_envelope(proposal_envelope, expected_schema=INITIAL_PROPOSAL_SCHEMA)
        gates.update(
            {
                "exactly_one_reset": proposal.get("reset_count") == 1,
                "exactly_one_policy_query": proposal.get("policy_query_count") == 1,
                "zero_environment_steps": proposal.get("environment_step_count") == 0,
                "zero_outcomes": proposal.get("outcomes_generated") is False,
                "proposal_binding_matches": (
                    proposal.get("policy_binding_digest") == binding_envelope["content_digest"]
                ),
            }
        )
    if candidate_envelope is not None:
        candidate = validate_envelope(candidate_envelope, expected_schema=CANDIDATE_REQUEST_SCHEMA)
        gates["four_raw_candidates"] = candidate.get("candidate_count") == 4
        gates["candidate_binding_matches"] = (
            candidate.get("policy_binding_digest") == binding_envelope["content_digest"]
        )
    if projected_envelope is not None:
        projected = validate_envelope(
            projected_envelope, expected_schema=PROJECTED_CANDIDATES_SCHEMA
        )
        gates["four_projected_candidates"] = projected.get("candidate_count") == 4
        gates["projection_zero_steps"] = projected.get("environment_step_count") == 0
        gates["projection_zero_outcomes"] = projected.get("outcomes_available") is False
        gates["projection_binding_matches"] = (
            projected.get("policy_binding_digest") == binding_envelope["content_digest"]
        )
        if candidate_envelope is not None:
            gates["projection_request_matches"] = projected.get(
                "raw_candidate_manifest_digest"
            ) == candidate_envelope["content_digest"] and projected.get(
                "candidate_pool_digest"
            ) == candidate.get("candidate_pool_digest")
            recomputed = project_candidates(
                binding_envelope=binding_envelope,
                candidate_envelope=candidate_envelope,
            )
            gates["projection_recomputes_exactly"] = (
                recomputed["content_digest"] == projected_envelope["content_digest"]
            )
    payload = {
        "gates": gates,
        "initial_state_only": True,
        "intermediate_policy_state_restoration": False,
        "passed": all(gates.values()),
        "physical_action_execution": False,
        "policy_binding_digest": binding_envelope["content_digest"],
    }
    return make_envelope(BRIDGE_AUDIT_SCHEMA, payload)


def load_runtime_registry(
    *, checkpoint_root: Path, dataset_root: Path, runtime_selection_path: Path
) -> tuple[ControllerRegistry, ControllerRegistryLocators]:
    """Load accepted controller metadata for a proposal command."""

    return load_controller_registry_metadata(
        checkpoint_root=checkpoint_root,
        dataset_root=dataset_root,
        runtime_selection_path=runtime_selection_path,
    )


__all__ = [
    "BRIDGE_AUDIT_SCHEMA",
    "CANDIDATE_REQUEST_SCHEMA",
    "INITIAL_PROPOSAL_SCHEMA",
    "POLICY_BINDING_SCHEMA",
    "PROJECTED_CANDIDATES_SCHEMA",
    "LatentGuardBridgeError",
    "audit_bridge",
    "build_policy_binding",
    "content_digest",
    "derive_probe_seed",
    "export_initial_proposal",
    "export_policy_binding",
    "load_runtime_registry",
    "make_envelope",
    "project_candidates",
    "read_envelope",
    "validate_envelope",
    "write_envelope",
]
