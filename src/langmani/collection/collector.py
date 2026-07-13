"""Deterministic, resumable M3A raw-demonstration collector."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from langmani.collection.manifest import (
    initialize_or_resume_manifest,
    persist_manifest,
)
from langmani.collection.recording import (
    NativeEpisodeLocation,
    close_recorder,
    create_candidate_recorder,
    flush_attempt,
    native_trajectory_paths,
)
from langmani.collection.replay import validate_action_replay
from langmani.datasets.archive import (
    ArchiveValidationError,
    EpisodeSelection,
    build_source_shard,
    create_counterfactual_scene_group_bundle,
    sha256_file,
    validate_native_archive_pair,
)
from langmani.datasets.identity import (
    stable_accepted_scene_group_id,
    stable_attempt_id,
    stable_raw_trajectory_id,
    stable_source_shard_id,
)
from langmani.datasets.schedule import scheduled_scene_group
from langmani.datasets.types import (
    AttemptOutcome,
    AttemptRecord,
    CollectionConfig,
    CollectionManifest,
    CollectionStatus,
    RawDatasetSummary,
    RawEpisodeRecord,
    ReplayValidationResult,
    SceneGroupRecord,
    ScheduledEpisode,
    SourceShardRecord,
)
from langmani.environments.specs import EpisodeSpec
from langmani.experts import PickPlaceExpert
from langmani.experts.command_support import (
    create_expert_environment,
    generic_unexpected_result,
    task_reset_options,
)
from langmani.experts.types import ExpertPhase, ExpertResult, ExpertStatus

RecorderFactory = Callable[[Path, str], Any]
ExpertFactory = Callable[[Any, CollectionConfig], Any]
ReplayValidator = Callable[..., ReplayValidationResult]

_INTERRUPTED_CANDIDATE_REASON = (
    "collector process interrupted before the candidate transaction was committed"
)
_EVALUATION_FIELDS = (
    "target_in_target_bin",
    "target_in_wrong_bin",
    "wrong_object_in_target_bin",
    "target_is_grasped",
    "target_is_static",
    "target_off_table",
    "success",
    "fail",
)
_PLANNED_PHASES = frozenset(
    {
        ExpertPhase.MOVE_TO_PREGRASP,
        ExpertPhase.APPROACH_TARGET,
        ExpertPhase.LIFT_TARGET,
        ExpertPhase.MOVE_ABOVE_DESTINATION,
        ExpertPhase.DESCEND_TO_PLACE,
        ExpertPhase.RETREAT,
    }
)


class CollectionError(RuntimeError):
    """Raised when collection cannot safely continue or persist evidence."""


class _CandidateStructuralRejection(RuntimeError):
    """Expected all-six structural rejection discovered during bundle admission."""


@dataclass(slots=True)
class _AttemptDraft:
    scheduled: ScheduledEpisode
    attempt_index: int
    attempt_id: str
    expert_result: ExpertResult
    native_location: NativeEpisodeLocation | None
    raw_trajectory_id: str | None
    base_outcome: AttemptOutcome | None
    failure_reasons: list[str]
    initial_scene_state: dict[str, Any] | None = None
    replay_validation: ReplayValidationResult | None = None
    selected: bool = False


@dataclass(slots=True)
class _CandidateResult:
    scheduled: tuple[ScheduledEpisode, ...]
    drafts: list[_AttemptDraft]
    selected: list[_AttemptDraft]
    accepted: bool
    rejection_reasons: list[str]
    candidate_directory: Path
    candidate_h5_path: Path
    has_retained_native_episodes: bool


def _default_recorder_factory(directory: Path, sim_backend: str) -> Any:
    return create_candidate_recorder(directory, sim_backend=sim_backend)


def _default_expert_factory(env: Any, config: CollectionConfig) -> PickPlaceExpert:
    return PickPlaceExpert(env, config=config.expert_config)


class RawDemonstrationCollector:
    """Collect complete six-task scene groups in deterministic candidate order."""

    def __init__(
        self,
        config: CollectionConfig,
        *,
        recorder_factory: RecorderFactory | None = None,
        expert_factory: ExpertFactory | None = None,
        replay_validator: ReplayValidator | None = None,
    ) -> None:
        if not isinstance(config, CollectionConfig):
            raise TypeError("config must be a CollectionConfig")
        self.config = config
        self.root = Path(config.raw_output_root).resolve()
        self._recorder_factory = recorder_factory or _default_recorder_factory
        self._expert_factory = expert_factory or _default_expert_factory
        self._replay_validator = replay_validator or validate_action_replay
        self._uses_default_replay_validator = replay_validator is None

    def collect(self) -> RawDatasetSummary:
        """Run or resume collection until the target or candidate bound is reached."""
        manifest = initialize_or_resume_manifest(self.config)
        manifest = self._reconcile_resume_state(manifest)
        if manifest.status is not CollectionStatus.IN_PROGRESS:
            return RawDatasetSummary.from_manifest(manifest)

        for candidate_index in range(
            manifest.next_candidate_scene_index,
            self.config.maximum_candidate_scene_count,
        ):
            if manifest.accepted_scene_group_count >= self.config.target_complete_scene_count:
                break
            candidate = self._collect_candidate(manifest, candidate_index)
            if candidate.accepted:
                try:
                    manifest = self._commit_accepted_candidate(manifest, candidate)
                except _CandidateStructuralRejection as error:
                    candidate.accepted = False
                    candidate.rejection_reasons.append(str(error))
                    candidate.rejection_reasons = list(dict.fromkeys(candidate.rejection_reasons))
                    self._write_inflight_candidate(
                        manifest,
                        scheduled=candidate.scheduled,
                        drafts=candidate.drafts,
                        rejection_reasons=candidate.rejection_reasons,
                        active_attempt=None,
                    )
                    manifest = self._commit_rejected_candidate(manifest, candidate)
            else:
                manifest = self._commit_rejected_candidate(manifest, candidate)

            reached_target = (
                manifest.accepted_scene_group_count == self.config.target_complete_scene_count
            )
            manifest = replace(
                manifest,
                status=(
                    CollectionStatus.COMPLETE if reached_target else CollectionStatus.IN_PROGRESS
                ),
            )
            persist_manifest(self.root, manifest)
            if candidate.accepted:
                retained = self._dispose_candidate(candidate, accepted=True)
                if retained != self._should_retain_candidate(candidate, accepted=True):
                    raise CollectionError(
                        "accepted candidate failure-retention result differs from manifest"
                    )
            self._cleanup_committed_transactions(manifest)
            if reached_target:
                return RawDatasetSummary.from_manifest(manifest)

        if manifest.accepted_scene_group_count < self.config.target_complete_scene_count:
            reason = (
                "candidate scene bound exhausted: accepted "
                f"{manifest.accepted_scene_group_count}/"
                f"{self.config.target_complete_scene_count} complete groups after "
                f"{manifest.next_candidate_scene_index}/"
                f"{self.config.maximum_candidate_scene_count} candidates"
            )
            manifest = replace(
                manifest,
                status=CollectionStatus.FAILED,
                failure_reasons=(reason,),
            )
            persist_manifest(self.root, manifest)
            self._cleanup_committed_transactions(manifest)
        return RawDatasetSummary.from_manifest(manifest)

    def _collect_candidate(
        self,
        manifest: CollectionManifest,
        candidate_index: int,
    ) -> _CandidateResult:
        scheduled = scheduled_scene_group(manifest.schedule, candidate_index)
        candidate = scheduled[0]
        candidate_directory = (
            self.root
            / ".staging"
            / f"candidate-{candidate_index:06d}-{_id_token(candidate.candidate_scene_id)}"
        )
        if candidate_directory.exists():
            shutil.rmtree(candidate_directory)
        self._write_inflight_candidate(
            manifest,
            scheduled=scheduled,
            drafts=(),
            rejection_reasons=(),
            active_attempt=None,
        )
        candidate_h5, _ = native_trajectory_paths(candidate_directory)
        drafts: list[_AttemptDraft] = []
        selected: list[_AttemptDraft] = []
        rejection_reasons: list[str] = []
        recorder: Any | None = None
        close_error: Exception | None = None

        try:
            recorder = self._recorder_factory(candidate_directory, self.config.sim_backend)
        except Exception as error:
            reason = f"candidate recorder initialization failed: {_error_text(error)}"
            rejection_reasons.append(reason)
            for episode in scheduled:
                drafts.append(self._unexpected_draft(episode, error))
            self._write_inflight_candidate(
                manifest,
                scheduled=scheduled,
                drafts=drafts,
                rejection_reasons=rejection_reasons,
                active_attempt=None,
            )
            return _CandidateResult(
                scheduled=scheduled,
                drafts=drafts,
                selected=[],
                accepted=False,
                rejection_reasons=rejection_reasons,
                candidate_directory=candidate_directory,
                candidate_h5_path=candidate_h5,
                has_retained_native_episodes=False,
            )

        recorder_usable = True
        try:
            for episode in scheduled:
                selected_for_task: _AttemptDraft | None = None
                for attempt_index in range(1, self.config.maximum_expert_attempts_per_task + 1):
                    if not recorder_usable:
                        break
                    attempt_id = stable_attempt_id(
                        scheduled_episode_id=episode.scheduled_episode_id,
                        attempt_index=attempt_index,
                    )
                    self._write_inflight_candidate(
                        manifest,
                        scheduled=scheduled,
                        drafts=drafts,
                        rejection_reasons=rejection_reasons,
                        active_attempt=(episode, attempt_index, attempt_id),
                    )
                    draft = self._run_attempt(
                        recorder,
                        episode=episode,
                        attempt_index=attempt_index,
                        candidate_directory=candidate_directory,
                    )
                    drafts.append(draft)
                    self._write_inflight_candidate(
                        manifest,
                        scheduled=scheduled,
                        drafts=drafts,
                        rejection_reasons=rejection_reasons,
                        active_attempt=None,
                    )
                    if draft.base_outcome is AttemptOutcome.RECORDING_FAILURE:
                        recorder_usable = False
                        rejection_reasons.extend(draft.failure_reasons)
                        break
                    if (
                        draft.base_outcome is None
                        and draft.expert_result.success
                        and draft.native_location is not None
                    ):
                        draft.selected = True
                        selected_for_task = draft
                        selected.append(draft)
                        break
                if selected_for_task is None:
                    rejection_reasons.append(
                        f"{episode.task_id} was not solved with a valid raw trajectory within "
                        f"{self.config.maximum_expert_attempts_per_task} attempt(s)"
                    )
        finally:
            try:
                close_recorder(recorder)
            except Exception as error:
                close_error = error

        if close_error is not None:
            rejection_reasons.append(f"candidate recorder close failed: {_error_text(close_error)}")

        has_native = candidate_h5.is_file() and candidate_h5.stat().st_size > 0
        if has_native:
            try:
                validate_native_archive_pair(candidate_h5)
            except Exception as error:
                rejection_reasons.append(
                    f"candidate native archive structural validation failed: {_error_text(error)}"
                )
        elif selected:
            rejection_reasons.append("candidate native archive pair was not persisted")

        can_replay = len(selected) == 6 and not rejection_reasons
        if can_replay:
            replay_env: Any | None = None
            try:
                if self._uses_default_replay_validator:
                    replay_env = create_expert_environment(
                        diagnostic_rendering=False,
                        sim_backend=self.config.sim_backend,
                    )
                for draft in selected:
                    assert draft.native_location is not None
                    replay_kwargs: dict[str, Any] = {}
                    if replay_env is not None:
                        replay_kwargs["environment"] = replay_env
                    try:
                        replay = self._replay_validator(
                            candidate_h5,
                            native_episode_id=draft.native_location.episode_id,
                            scheduled_episode=draft.scheduled,
                            config=self.config,
                            sim_backend=self.config.sim_backend,
                            **replay_kwargs,
                        )
                    except Exception as error:
                        draft.failure_reasons.append(
                            f"replay validation raised {_error_text(error)}"
                        )
                        draft.base_outcome = AttemptOutcome.REPLAY_FAILURE
                        rejection_reasons.extend(draft.failure_reasons)
                        self._write_inflight_candidate(
                            manifest,
                            scheduled=scheduled,
                            drafts=drafts,
                            rejection_reasons=rejection_reasons,
                            active_attempt=None,
                        )
                        continue
                    draft.replay_validation = replay
                    if not replay.passed:
                        draft.failure_reasons.extend(replay.failure_reasons)
                        rejection_reasons.append(
                            f"{draft.scheduled.task_id} failed action replay validation"
                        )
                        rejection_reasons.extend(replay.failure_reasons)
                    self._write_inflight_candidate(
                        manifest,
                        scheduled=scheduled,
                        drafts=drafts,
                        rejection_reasons=rejection_reasons,
                        active_attempt=None,
                    )
            finally:
                if replay_env is not None:
                    try:
                        replay_env.close()
                    except Exception as error:
                        rejection_reasons.append(
                            f"shared replay environment close failed: {_error_text(error)}"
                        )

        accepted = (
            len(selected) == 6
            and not rejection_reasons
            and all(
                draft.replay_validation is not None and draft.replay_validation.passed
                for draft in selected
            )
        )
        self._write_inflight_candidate(
            manifest,
            scheduled=scheduled,
            drafts=drafts,
            rejection_reasons=rejection_reasons,
            active_attempt=None,
        )
        return _CandidateResult(
            scheduled=scheduled,
            drafts=drafts,
            selected=selected,
            accepted=accepted,
            rejection_reasons=list(dict.fromkeys(rejection_reasons)),
            candidate_directory=candidate_directory,
            candidate_h5_path=candidate_h5,
            has_retained_native_episodes=has_native,
        )

    def _run_attempt(
        self,
        recorder: Any,
        *,
        episode: ScheduledEpisode,
        attempt_index: int,
        candidate_directory: Path,
    ) -> _AttemptDraft:
        attempt_id = stable_attempt_id(
            scheduled_episode_id=episode.scheduled_episode_id,
            attempt_index=attempt_index,
        )
        expert: Any | None = None
        expert_result: ExpertResult | None = None
        reset_episode_spec: EpisodeSpec | None = None
        actual_evaluation: dict[str, bool] | None = None
        initial_scene_state: dict[str, Any] | None = None
        reasons: list[str] = []
        try:
            recorder.reset(
                seed=episode.scene_seed,
                options=task_reset_options(episode.task_spec),
            )
            metadata_env = _metadata_environment(recorder)
            try:
                reset_specs = metadata_env.get_episode_specs()
                if not isinstance(reset_specs, tuple) or len(reset_specs) != 1:
                    raise CollectionError(
                        "reset metadata accessor must return exactly one EpisodeSpec"
                    )
                reset_episode_spec = reset_specs[0]
                if not isinstance(reset_episode_spec, EpisodeSpec):
                    raise CollectionError("reset metadata accessor did not return an EpisodeSpec")
                initial_scene_state = _normalize_initial_scene_state(
                    metadata_env.get_expert_initial_scene_state()
                )
            except Exception as error:
                reasons.append(
                    "reset metadata capture failed before expert execution: " + _error_text(error)
                )
            expert = self._expert_factory(recorder, self.config)
            expert_result = expert.run()
        except Exception as error:
            expert_result = (
                expert.unexpected_exception_result(error)
                if expert is not None
                else generic_unexpected_result(
                    error,
                    scene_seed=episode.scene_seed,
                    task_spec=episode.task_spec,
                )
            )

        if expert_result.success:
            try:
                actual_evaluation = _materialize_evaluation(
                    _metadata_environment(recorder).get_expert_evaluation()
                )
            except Exception as error:
                reasons.append("final environment evaluation capture failed: " + _error_text(error))

        reasons.extend(
            _expert_contract_failures(
                expert_result,
                scheduled=episode,
                reset_episode_spec=reset_episode_spec,
                actual_evaluation=actual_evaluation,
                attempt_index=attempt_index,
                config=self.config,
            )
        )

        should_save = expert_result.success or self.config.retain_failed_raw_trajectories
        location: NativeEpisodeLocation | None = None
        base_outcome: AttemptOutcome | None = None
        try:
            location = flush_attempt(
                recorder,
                save=should_save,
                output_directory=candidate_directory,
            )
        except Exception as error:
            reasons.append(f"native trajectory flush failed: {_error_text(error)}")
            base_outcome = AttemptOutcome.RECORDING_FAILURE

        raw_id = (
            stable_raw_trajectory_id(
                scheduled_episode_id=episode.scheduled_episode_id,
                attempt_id=attempt_id,
            )
            if location is not None
            else None
        )
        if expert_result.success and location is None and base_outcome is None:
            reasons.append("successful expert attempt produced no native trajectory")
            base_outcome = AttemptOutcome.RECORDING_FAILURE
        if expert_result.success and reasons and base_outcome is None:
            base_outcome = AttemptOutcome.EXPERT_CONTRACT_FAILURE
        if not expert_result.success:
            if base_outcome is None:
                base_outcome = (
                    AttemptOutcome.UNEXPECTED_EXCEPTION
                    if expert_result.status is ExpertStatus.UNEXPECTED_EXCEPTION
                    else AttemptOutcome.EXPERT_FAILURE
                )
            reasons.append(f"expert finished with status={expert_result.status.value}")
        return _AttemptDraft(
            scheduled=episode,
            attempt_index=attempt_index,
            attempt_id=attempt_id,
            expert_result=expert_result,
            native_location=location,
            raw_trajectory_id=raw_id,
            base_outcome=base_outcome,
            failure_reasons=reasons,
            initial_scene_state=initial_scene_state,
        )

    def _unexpected_draft(
        self,
        episode: ScheduledEpisode,
        error: Exception,
    ) -> _AttemptDraft:
        attempt_id = stable_attempt_id(
            scheduled_episode_id=episode.scheduled_episode_id,
            attempt_index=1,
        )
        return _AttemptDraft(
            scheduled=episode,
            attempt_index=1,
            attempt_id=attempt_id,
            expert_result=generic_unexpected_result(
                error,
                scene_seed=episode.scene_seed,
                task_spec=episode.task_spec,
            ),
            native_location=None,
            raw_trajectory_id=None,
            base_outcome=AttemptOutcome.UNEXPECTED_EXCEPTION,
            failure_reasons=[f"candidate initialization failed: {_error_text(error)}"],
        )

    def _commit_rejected_candidate(
        self,
        manifest: CollectionManifest,
        candidate: _CandidateResult,
    ) -> CollectionManifest:
        retained = self._dispose_candidate(candidate, accepted=False)
        attempts = tuple(
            self._final_attempt_record(
                draft,
                group_accepted=False,
                candidate_reasons=candidate.rejection_reasons,
                retained_failure_pair=retained,
            )
            for draft in candidate.drafts
        )
        first = candidate.scheduled[0]
        group = SceneGroupRecord(
            candidate_scene_index=first.candidate_scene_index,
            candidate_scene_id=first.candidate_scene_id,
            accepted_scene_group_id=None,
            scene_seed=first.scene_seed,
            scene_id=first.scene_id,
            scheduled_episode_ids=tuple(
                episode.scheduled_episode_id for episode in candidate.scheduled
            ),
            attempt_ids=tuple(attempt.attempt_id for attempt in attempts),
            episodes=(),
            complete=False,
            accepted=False,
            rejection_reasons=tuple(candidate.rejection_reasons or ["candidate group rejected"]),
        )
        return replace(
            manifest,
            next_candidate_scene_index=first.candidate_scene_index + 1,
            attempts=manifest.attempts + attempts,
            scene_groups=manifest.scene_groups + (group,),
        )

    def _commit_accepted_candidate(
        self,
        manifest: CollectionManifest,
        candidate: _CandidateResult,
    ) -> CollectionManifest:
        raw_ids = tuple(_require_raw_id(draft) for draft in candidate.selected)
        group_id = stable_accepted_scene_group_id(
            candidate_scene_id=candidate.scheduled[0].candidate_scene_id,
            raw_trajectory_ids=raw_ids,
        )
        accepted_index = manifest.accepted_scene_group_count
        shard_index = (accepted_index * 6) // self.config.shard_size
        shard_id = stable_source_shard_id(
            collection_run_id=manifest.collection_run_id,
            shard_index=shard_index,
        )
        bundle_path = self._group_bundle_path(shard_index, group_id)
        selections = tuple(
            EpisodeSelection(
                native_episode_id=_require_location(draft).episode_id,
                langmani_metadata=self._accepted_episode_metadata(
                    manifest,
                    draft,
                    group_id=group_id,
                    shard_id=shard_id,
                ),
            )
            for draft in candidate.selected
        )
        try:
            bundle_validation = create_counterfactual_scene_group_bundle(
                candidate.candidate_h5_path,
                selections,
                bundle_path,
                group_metadata={
                    "collection_schema_version": manifest.collection_schema_version,
                    "collection_run_id": manifest.collection_run_id,
                    "candidate_scene_id": candidate.scheduled[0].candidate_scene_id,
                    "accepted_scene_group_id": group_id,
                    "scene_seed": candidate.scheduled[0].scene_seed,
                    "scene_id": candidate.scheduled[0].scene_id,
                    "expert_config_fingerprint": self.config.expert_config_fingerprint,
                },
            )
        except ArchiveValidationError as error:
            raise _CandidateStructuralRejection(
                "counterfactual structural validation failed: " + _error_text(error)
            ) from error
        retained_failures = self._should_retain_candidate(candidate, accepted=True)
        new_attempts = tuple(
            self._final_attempt_record(
                draft,
                group_accepted=True,
                candidate_reasons=(),
                retained_failure_pair=retained_failures,
            )
            for draft in candidate.drafts
        )
        return self._publish_shard_and_group(
            manifest,
            candidate=candidate,
            group_id=group_id,
            shard_id=shard_id,
            shard_index=shard_index,
            new_attempts=new_attempts,
            bundle_h5_sha256=bundle_validation.h5_sha256,
            bundle_json_sha256=bundle_validation.json_sha256,
        )

    def _publish_shard_and_group(
        self,
        manifest: CollectionManifest,
        *,
        candidate: _CandidateResult,
        group_id: str,
        shard_id: str,
        shard_index: int,
        new_attempts: tuple[AttemptRecord, ...],
        bundle_h5_sha256: str,
        bundle_json_sha256: str,
    ) -> CollectionManifest:
        existing_groups = tuple(
            group
            for group in manifest.scene_groups
            if group.accepted and group.episodes and group.episodes[0].source_shard_id == shard_id
        )
        group_ids = tuple(group.accepted_scene_group_id for group in existing_groups) + (group_id,)
        if any(value is None for value in group_ids):
            raise CollectionError("accepted scene group is missing its stable ID")
        bundle_paths = [self._group_bundle_path(shard_index, str(value)) for value in group_ids]
        if not all(path.is_file() for path in bundle_paths):
            missing = [str(path) for path in bundle_paths if not path.is_file()]
            raise CollectionError(f"cannot rebuild source shard; missing group bundles: {missing}")
        trusted_bundle_digests = [
            (
                group.accepted_scene_group_id or "",
                group.bundle_h5_sha256,
                group.bundle_json_sha256,
            )
            for group in existing_groups
        ] + [(group_id, bundle_h5_sha256, bundle_json_sha256)]
        for bundle_path, (trusted_group_id, h5_digest, json_digest) in zip(
            bundle_paths,
            trusted_bundle_digests,
            strict=True,
        ):
            self._require_trusted_group_bundle(
                bundle_path,
                group_id=trusted_group_id,
                h5_sha256=h5_digest,
                json_sha256=json_digest,
            )

        build_directory = self.root / ".staging" / "shard-builds"
        build_directory.mkdir(parents=True, exist_ok=True)
        build_path = build_directory / f"shard-{shard_index:05d}-{len(group_ids):03d}.h5"
        _remove_pair(build_path)
        build_source_shard(
            bundle_paths,
            build_path,
            shard_metadata={
                "collection_schema_version": manifest.collection_schema_version,
                "collection_run_id": manifest.collection_run_id,
                "source_shard_id": shard_id,
                "shard_index": shard_index,
                "expert_config_fingerprint": self.config.expert_config_fingerprint,
            },
        )
        canonical_h5 = self._source_shard_path(shard_index, shard_id)
        canonical_json = canonical_h5.with_suffix(".json")
        canonical_h5.parent.mkdir(parents=True, exist_ok=True)
        os.replace(build_path, canonical_h5)
        os.replace(build_path.with_suffix(".json"), canonical_json)
        validation = validate_native_archive_pair(
            canonical_h5,
            require_accepted_success=True,
            require_langmani_metadata=True,
        )

        episode_json = _episodes_by_id(canonical_json)
        old_records = {record.raw_trajectory_id: record for record in manifest.raw_episodes}
        new_drafts = {_require_raw_id(draft): draft for draft in candidate.selected}
        shard_records: list[RawEpisodeRecord] = []
        relative_h5 = canonical_h5.relative_to(self.root).as_posix()
        relative_json = canonical_json.relative_to(self.root).as_posix()
        for location in validation.episodes:
            raw_id = location.raw_trajectory_id
            if raw_id is None:
                raise CollectionError("source shard episode is missing raw_trajectory_id")
            if raw_id in new_drafts:
                draft = new_drafts[raw_id]
                scheduled = draft.scheduled
                expert_result = draft.expert_result
                replay = draft.replay_validation
                if replay is None:
                    raise CollectionError("accepted attempt is missing replay evidence")
            elif raw_id in old_records:
                old = old_records[raw_id]
                scheduled = next(
                    episode
                    for episode in manifest.schedule.episodes
                    if episode.scheduled_episode_id == old.scheduled_episode_id
                )
                expert_result = old.expert_result
                replay = old.replay_validation
            else:
                raise CollectionError(f"source shard contains unknown raw trajectory {raw_id}")
            native_metadata = episode_json[location.native_episode_id]
            shard_records.append(
                RawEpisodeRecord(
                    scheduled_episode_id=scheduled.scheduled_episode_id,
                    attempt_id=location.attempt_id
                    or _attempt_for_raw(raw_id, new_attempts, manifest),
                    raw_trajectory_id=raw_id,
                    source_shard_id=shard_id,
                    native_episode_id=location.native_episode_id,
                    h5_group=location.h5_group,
                    h5_path=relative_h5,
                    json_path=relative_json,
                    h5_sha256=validation.h5_sha256,
                    json_sha256=validation.json_sha256,
                    elapsed_steps=int(native_metadata["elapsed_steps"]),
                    scene_seed=scheduled.scene_seed,
                    scene_id=scheduled.scene_id,
                    task_spec=scheduled.task_spec,
                    task_id=scheduled.task_id,
                    canonical_instruction=scheduled.canonical_instruction,
                    expert_result=expert_result,
                    replay_validation=replay,
                    final_environment_evaluation=expert_result.final_environment_evaluation,
                    structural_valid=True,
                    accepted=True,
                )
            )

        records_by_raw = {record.raw_trajectory_id: record for record in shard_records}
        updated_existing_groups = tuple(
            replace(
                group,
                episodes=tuple(
                    records_by_raw[episode.raw_trajectory_id] for episode in group.episodes
                ),
            )
            if group in existing_groups
            else group
            for group in manifest.scene_groups
        )
        first = candidate.scheduled[0]
        new_group = SceneGroupRecord(
            candidate_scene_index=first.candidate_scene_index,
            candidate_scene_id=first.candidate_scene_id,
            accepted_scene_group_id=group_id,
            scene_seed=first.scene_seed,
            scene_id=first.scene_id,
            scheduled_episode_ids=tuple(
                episode.scheduled_episode_id for episode in candidate.scheduled
            ),
            attempt_ids=tuple(attempt.attempt_id for attempt in new_attempts),
            episodes=tuple(
                records_by_raw[raw_id]
                for raw_id in (_require_raw_id(draft) for draft in candidate.selected)
            ),
            complete=True,
            accepted=True,
            bundle_h5_sha256=bundle_h5_sha256,
            bundle_json_sha256=bundle_json_sha256,
        )
        unaffected_records = tuple(
            record for record in manifest.raw_episodes if record.source_shard_id != shard_id
        )
        shard_record = SourceShardRecord(
            source_shard_id=shard_id,
            shard_index=shard_index,
            h5_path=relative_h5,
            json_path=relative_json,
            h5_sha256=validation.h5_sha256,
            json_sha256=validation.json_sha256,
            h5_size_bytes=validation.h5_size_bytes,
            json_size_bytes=validation.json_size_bytes,
            episode_count=len(shard_records),
            raw_trajectory_ids=tuple(record.raw_trajectory_id for record in shard_records),
        )
        other_shards = tuple(
            shard for shard in manifest.source_shards if shard.source_shard_id != shard_id
        )
        updated = replace(
            manifest,
            next_candidate_scene_index=first.candidate_scene_index + 1,
            attempts=manifest.attempts + new_attempts,
            raw_episodes=unaffected_records + tuple(shard_records),
            scene_groups=updated_existing_groups + (new_group,),
            source_shards=tuple(
                sorted(other_shards + (shard_record,), key=lambda item: item.shard_index)
            ),
        )
        return updated

    def _accepted_episode_metadata(
        self,
        manifest: CollectionManifest,
        draft: _AttemptDraft,
        *,
        group_id: str,
        shard_id: str,
    ) -> dict[str, Any]:
        replay = draft.replay_validation
        if replay is None or not replay.passed:
            raise CollectionError("accepted episode metadata requires passed replay")
        return {
            "collection_schema_version": manifest.collection_schema_version,
            "collection_run_id": manifest.collection_run_id,
            "candidate_scene_id": draft.scheduled.candidate_scene_id,
            "accepted_scene_group_id": group_id,
            "source_shard_id": shard_id,
            "scheduled_episode_id": draft.scheduled.scheduled_episode_id,
            "attempt_id": draft.attempt_id,
            "raw_trajectory_id": _require_raw_id(draft),
            "scene_seed": draft.scheduled.scene_seed,
            "scene_id": draft.scheduled.scene_id,
            "task_spec": draft.scheduled.task_spec.to_dict(),
            "task_id": draft.scheduled.task_id,
            "canonical_instruction": draft.scheduled.canonical_instruction,
            "expert_config_fingerprint": self.config.expert_config_fingerprint,
            "expert_result": draft.expert_result.to_dict(),
            "replay_validation": replay.to_dict(),
            "final_environment_evaluation": dict(draft.expert_result.final_environment_evaluation),
            "initial_scene_state": _require_initial_scene_state(draft),
            "structural_valid": True,
            "accepted": True,
        }

    def _final_attempt_record(
        self,
        draft: _AttemptDraft,
        *,
        group_accepted: bool,
        candidate_reasons: tuple[str, ...] | list[str],
        retained_failure_pair: bool,
    ) -> AttemptRecord:
        if group_accepted and draft.selected:
            outcome = AttemptOutcome.ACCEPTED
            reasons: tuple[str, ...] = ()
            retained = True
        else:
            outcome = draft.base_outcome
            reasons_list = list(draft.failure_reasons)
            if outcome is None:
                if draft.replay_validation is not None and not draft.replay_validation.passed:
                    outcome = AttemptOutcome.REPLAY_FAILURE
                elif any("structural" in reason for reason in candidate_reasons):
                    outcome = AttemptOutcome.STRUCTURAL_FAILURE
                else:
                    outcome = AttemptOutcome.GROUP_REJECTED
            reasons_list.extend(candidate_reasons)
            reasons = tuple(dict.fromkeys(reasons_list or ["counterfactual scene group rejected"]))
            retained = retained_failure_pair and draft.native_location is not None
        return AttemptRecord(
            attempt_id=draft.attempt_id,
            scheduled_episode_id=draft.scheduled.scheduled_episode_id,
            attempt_index=draft.attempt_index,
            scene_seed=draft.scheduled.scene_seed,
            scene_id=draft.scheduled.scene_id,
            task_spec=draft.scheduled.task_spec,
            task_id=draft.scheduled.task_id,
            outcome=outcome,
            expert_result=draft.expert_result,
            raw_trajectory_id=draft.raw_trajectory_id,
            replay_validation=draft.replay_validation,
            trajectory_retained=retained,
            failure_reasons=reasons,
        )

    def _write_inflight_candidate(
        self,
        manifest: CollectionManifest,
        *,
        scheduled: tuple[ScheduledEpisode, ...],
        drafts: tuple[_AttemptDraft, ...] | list[_AttemptDraft],
        rejection_reasons: tuple[str, ...] | list[str],
        active_attempt: tuple[ScheduledEpisode, int, str] | None,
    ) -> None:
        """Atomically preserve every started/completed attempt before group commit."""
        first = scheduled[0]
        recovery_reasons = tuple(dict.fromkeys((*rejection_reasons, _INTERRUPTED_CANDIDATE_REASON)))
        provisional_attempts = tuple(
            self._final_attempt_record(
                draft,
                group_accepted=False,
                candidate_reasons=recovery_reasons,
                retained_failure_pair=False,
            )
            for draft in drafts
        )
        active_payload = None
        if active_attempt is not None:
            episode, attempt_index, attempt_id = active_attempt
            active_payload = {
                "scheduled_episode_id": episode.scheduled_episode_id,
                "attempt_index": attempt_index,
                "attempt_id": attempt_id,
            }
        payload = {
            "collection_schema_version": manifest.collection_schema_version,
            "collection_run_id": manifest.collection_run_id,
            "candidate_scene_index": first.candidate_scene_index,
            "candidate_scene_id": first.candidate_scene_id,
            "scene_seed": first.scene_seed,
            "scene_id": first.scene_id,
            "scheduled_episode_ids": [episode.scheduled_episode_id for episode in scheduled],
            "attempts": [attempt.to_dict() for attempt in provisional_attempts],
            "active_attempt": active_payload,
            "rejection_reasons": list(rejection_reasons),
        }
        _write_json_atomic(self._inflight_candidate_path(first.candidate_scene_index), payload)

    def _recover_inflight_candidates(self, manifest: CollectionManifest) -> CollectionManifest:
        """Turn an interrupted candidate into an explicit rejected group exactly once."""
        inflight_root = self.root / "journal" / "inflight"
        if not inflight_root.exists():
            return manifest
        paths = sorted(inflight_root.glob("candidate-*.json"))
        if len(paths) > 1:
            raise CollectionError("multiple in-flight candidates violate serial collection")
        for path in paths:
            payload = _read_json_mapping(path)
            expected_fields = {
                "collection_schema_version",
                "collection_run_id",
                "candidate_scene_index",
                "candidate_scene_id",
                "scene_seed",
                "scene_id",
                "scheduled_episode_ids",
                "attempts",
                "active_attempt",
                "rejection_reasons",
            }
            if set(payload) != expected_fields:
                raise CollectionError(f"malformed in-flight candidate journal: {path}")
            if payload["collection_schema_version"] != manifest.collection_schema_version:
                raise CollectionError("in-flight journal schema differs from manifest")
            if payload["collection_run_id"] != manifest.collection_run_id:
                raise CollectionError("in-flight journal belongs to another collection run")
            candidate_index = payload["candidate_scene_index"]
            if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
                raise CollectionError("in-flight candidate index must be an integer")
            if candidate_index < manifest.next_candidate_scene_index:
                self._finalize_committed_candidate_directory(manifest, candidate_index)
                path.unlink()
                continue
            if candidate_index != manifest.next_candidate_scene_index:
                raise CollectionError("in-flight candidate is not the next scheduled scene")

            scheduled = scheduled_scene_group(manifest.schedule, candidate_index)
            first = scheduled[0]
            expected_scheduled_ids = tuple(episode.scheduled_episode_id for episode in scheduled)
            if (
                payload["candidate_scene_id"] != first.candidate_scene_id
                or payload["scene_seed"] != first.scene_seed
                or payload["scene_id"] != first.scene_id
                or tuple(payload["scheduled_episode_ids"]) != expected_scheduled_ids
            ):
                raise CollectionError("in-flight journal differs from deterministic schedule")
            raw_attempts = payload["attempts"]
            if not isinstance(raw_attempts, list):
                raise CollectionError("in-flight attempts must be a list")
            try:
                attempts = tuple(
                    AttemptRecord.from_dict(item)
                    for item in raw_attempts
                    if isinstance(item, Mapping)
                )
            except (TypeError, ValueError, KeyError) as error:
                raise CollectionError(f"invalid in-flight attempt record: {error}") from error
            if len(attempts) != len(raw_attempts):
                raise CollectionError("in-flight attempt entries must be mappings")
            scheduled_by_id = {episode.scheduled_episode_id: episode for episode in scheduled}
            for attempt in attempts:
                episode = scheduled_by_id.get(attempt.scheduled_episode_id)
                if episode is None:
                    raise CollectionError("in-flight attempt references another candidate")
                if attempt.attempt_index > self.config.maximum_expert_attempts_per_task:
                    raise CollectionError("in-flight attempt exceeds the configured retry bound")
                expected_attempt_id = stable_attempt_id(
                    scheduled_episode_id=episode.scheduled_episode_id,
                    attempt_index=attempt.attempt_index,
                )
                if (
                    attempt.attempt_id != expected_attempt_id
                    or attempt.scene_seed != episode.scene_seed
                    or attempt.scene_id != episode.scene_id
                    or attempt.task_spec != episode.task_spec
                    or attempt.task_id != episode.task_id
                ):
                    raise CollectionError("in-flight attempt provenance differs from schedule")

            active = payload["active_attempt"]
            if active is not None:
                if not isinstance(active, Mapping) or set(active) != {
                    "scheduled_episode_id",
                    "attempt_index",
                    "attempt_id",
                }:
                    raise CollectionError("malformed active attempt journal entry")
                scheduled_id = active["scheduled_episode_id"]
                attempt_index = active["attempt_index"]
                attempt_id = active["attempt_id"]
                episode = scheduled_by_id.get(scheduled_id)
                if (
                    episode is None
                    or isinstance(attempt_index, bool)
                    or not isinstance(attempt_index, int)
                    or attempt_index < 1
                    or attempt_index > self.config.maximum_expert_attempts_per_task
                    or attempt_id
                    != stable_attempt_id(
                        scheduled_episode_id=episode.scheduled_episode_id,
                        attempt_index=attempt_index,
                    )
                    or any(record.attempt_id == attempt_id for record in attempts)
                ):
                    raise CollectionError("active attempt journal entry is inconsistent")
                interruption = RuntimeError(_INTERRUPTED_CANDIDATE_REASON)
                interrupted_draft = _AttemptDraft(
                    scheduled=episode,
                    attempt_index=attempt_index,
                    attempt_id=attempt_id,
                    expert_result=generic_unexpected_result(
                        interruption,
                        scene_seed=episode.scene_seed,
                        task_spec=episode.task_spec,
                    ),
                    native_location=None,
                    raw_trajectory_id=None,
                    base_outcome=AttemptOutcome.UNEXPECTED_EXCEPTION,
                    failure_reasons=[_INTERRUPTED_CANDIDATE_REASON],
                )
                attempts += (
                    self._final_attempt_record(
                        interrupted_draft,
                        group_accepted=False,
                        candidate_reasons=(_INTERRUPTED_CANDIDATE_REASON,),
                        retained_failure_pair=False,
                    ),
                )

            existing_attempt_ids = {attempt.attempt_id for attempt in manifest.attempts}
            recovered_attempt_ids = [attempt.attempt_id for attempt in attempts]
            if len(set(recovered_attempt_ids)) != len(
                recovered_attempt_ids
            ) or existing_attempt_ids.intersection(recovered_attempt_ids):
                raise CollectionError("in-flight attempt identity is duplicated")
            retained = self._recover_interrupted_candidate_directory(scheduled)
            if retained:
                attempts = tuple(
                    replace(
                        attempt,
                        trajectory_retained=attempt.raw_trajectory_id is not None,
                    )
                    for attempt in attempts
                )
            rejection_reasons = payload["rejection_reasons"]
            if not isinstance(rejection_reasons, list) or not all(
                isinstance(reason, str) and reason for reason in rejection_reasons
            ):
                raise CollectionError("in-flight rejection reasons must be strings")
            reasons = tuple(dict.fromkeys((*rejection_reasons, _INTERRUPTED_CANDIDATE_REASON)))
            group = SceneGroupRecord(
                candidate_scene_index=candidate_index,
                candidate_scene_id=first.candidate_scene_id,
                accepted_scene_group_id=None,
                scene_seed=first.scene_seed,
                scene_id=first.scene_id,
                scheduled_episode_ids=expected_scheduled_ids,
                attempt_ids=tuple(attempt.attempt_id for attempt in attempts),
                episodes=(),
                complete=False,
                accepted=False,
                rejection_reasons=reasons,
            )
            manifest = replace(
                manifest,
                next_candidate_scene_index=candidate_index + 1,
                attempts=manifest.attempts + attempts,
                scene_groups=manifest.scene_groups + (group,),
            )
        return manifest

    def _recover_interrupted_candidate_directory(
        self,
        scheduled: tuple[ScheduledEpisode, ...],
    ) -> bool:
        first = scheduled[0]
        staging = (
            self.root
            / ".staging"
            / f"candidate-{first.candidate_scene_index:06d}-{_id_token(first.candidate_scene_id)}"
        )
        destination = (
            self.root
            / "failed"
            / f"candidate-{first.candidate_scene_index:06d}-{_id_token(first.candidate_scene_id)}"
        )
        if destination.exists() and staging.exists():
            raise CollectionError("interrupted candidate exists in both staging and failure areas")
        if self.config.retain_failed_raw_trajectories:
            if staging.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, destination)
            return (
                destination.joinpath("attempts.h5").is_file()
                and destination.joinpath("attempts.json").is_file()
            )
        if destination.exists():
            raise CollectionError("unexpected retained failure directory for current configuration")
        if staging.exists():
            shutil.rmtree(staging)
        return False

    def _finalize_committed_candidate_directory(
        self,
        manifest: CollectionManifest,
        candidate_index: int,
    ) -> None:
        """Idempotently finish post-manifest staging disposal after a crash."""
        group = manifest.scene_groups[candidate_index]
        scheduled = scheduled_scene_group(manifest.schedule, candidate_index)
        first = scheduled[0]
        staging = (
            self.root
            / ".staging"
            / f"candidate-{candidate_index:06d}-{_id_token(first.candidate_scene_id)}"
        )
        destination = (
            self.root
            / "failed"
            / f"candidate-{candidate_index:06d}-{_id_token(first.candidate_scene_id)}"
        )
        attempts_by_id = {attempt.attempt_id: attempt for attempt in manifest.attempts}
        attempts = tuple(attempts_by_id[attempt_id] for attempt_id in group.attempt_ids)
        retain_failure_pair = any(
            attempt.outcome is not AttemptOutcome.ACCEPTED and attempt.trajectory_retained
            for attempt in attempts
        )
        if retain_failure_pair:
            if staging.exists() and destination.exists():
                raise CollectionError(
                    "committed candidate exists in both staging and failure areas"
                )
            if staging.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, destination)
            if not destination.is_dir():
                raise CollectionError(
                    "manifest declares retained failed trajectories but their directory is absent"
                )
            failure_index_path = destination / "failure_index.json"
            if not failure_index_path.exists():
                _write_json_atomic(
                    failure_index_path,
                    {
                        "candidate_scene_id": group.candidate_scene_id,
                        "scene_seed": group.scene_seed,
                        "accepted_scene_group": group.accepted,
                        "rejection_reasons": list(group.rejection_reasons),
                        "attempts": [attempt.to_dict() for attempt in attempts],
                        "recovered_after_manifest_commit": True,
                    },
                )
            return
        if destination.exists():
            raise CollectionError(
                "failure directory exists although the manifest does not retain failed raw data"
            )
        if staging.exists():
            shutil.rmtree(staging)

    def _dispose_candidate(self, candidate: _CandidateResult, *, accepted: bool) -> bool:
        retain = self._should_retain_candidate(candidate, accepted=accepted)
        if retain and candidate.candidate_directory.exists():
            destination = (
                self.root
                / "failed"
                / f"candidate-{candidate.scheduled[0].candidate_scene_index:06d}-"
                f"{_id_token(candidate.scheduled[0].candidate_scene_id)}"
            )
            if destination.exists():
                raise CollectionError(
                    f"failure retention destination already exists: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(candidate.candidate_directory, destination)
            failure_index = {
                "candidate_scene_id": candidate.scheduled[0].candidate_scene_id,
                "scene_seed": candidate.scheduled[0].scene_seed,
                "accepted_scene_group": accepted,
                "rejection_reasons": candidate.rejection_reasons,
                "attempts": [
                    {
                        "attempt_id": draft.attempt_id,
                        "raw_trajectory_id": draft.raw_trajectory_id,
                        "native_episode_id": (
                            draft.native_location.episode_id
                            if draft.native_location is not None
                            else None
                        ),
                        "selected_for_accepted_group": accepted and draft.selected,
                        "expert_result": draft.expert_result.to_dict(),
                    }
                    for draft in candidate.drafts
                ],
            }
            (destination / "failure_index.json").write_text(
                json.dumps(failure_index, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return True
        if candidate.candidate_directory.exists():
            shutil.rmtree(candidate.candidate_directory)
        return False

    def _should_retain_candidate(
        self,
        candidate: _CandidateResult,
        *,
        accepted: bool,
    ) -> bool:
        failed_raw_exists = any(
            draft.native_location is not None and not (accepted and draft.selected)
            for draft in candidate.drafts
        )
        return self.config.retain_failed_raw_trajectories and (failed_raw_exists or not accepted)

    def _reconcile_resume_state(self, manifest: CollectionManifest) -> CollectionManifest:
        manifest = self._recover_inflight_candidates(manifest)
        staging = self.root / ".staging"
        if staging.exists():
            shutil.rmtree(staging)

        expected_accepted_h5 = {
            (self.root / shard.h5_path).resolve() for shard in manifest.source_shards
        }
        accepted_root = self.root / "accepted" / "shards"
        if accepted_root.exists():
            for path in accepted_root.glob("*.h5"):
                if path.resolve() not in expected_accepted_h5:
                    _remove_pair(path)
            expected_accepted_json = {
                (self.root / shard.json_path).resolve() for shard in manifest.source_shards
            }
            for path in accepted_root.glob("*.json"):
                if path.resolve() not in expected_accepted_json:
                    path.unlink()

        journal_root = self.root / "journal"
        terminal = manifest.status is not CollectionStatus.IN_PROGRESS
        expected_bundles = {
            self._group_bundle_path(shard.shard_index, group.accepted_scene_group_id)
            for shard in manifest.source_shards
            if not terminal and shard.episode_count < self.config.shard_size
            for group in manifest.scene_groups
            if group.accepted
            and group.accepted_scene_group_id is not None
            and group.episodes
            and group.episodes[0].source_shard_id == shard.source_shard_id
        }
        if journal_root.exists():
            for path in journal_root.rglob("*.h5"):
                if path.resolve() not in {item.resolve() for item in expected_bundles}:
                    _remove_pair(path)
        if not terminal:
            for shard in manifest.source_shards:
                if shard.episode_count >= self.config.shard_size:
                    continue
                for group in manifest.scene_groups:
                    if (
                        group.accepted
                        and group.accepted_scene_group_id is not None
                        and group.episodes
                        and group.episodes[0].source_shard_id == shard.source_shard_id
                    ):
                        self._require_trusted_group_bundle(
                            self._group_bundle_path(
                                shard.shard_index,
                                group.accepted_scene_group_id,
                            ),
                            group_id=group.accepted_scene_group_id,
                            h5_sha256=group.bundle_h5_sha256,
                            json_sha256=group.bundle_json_sha256,
                        )
        for shard in manifest.source_shards:
            h5_path = self.root / shard.h5_path
            try:
                validation = validate_native_archive_pair(
                    h5_path,
                    require_accepted_success=True,
                    require_langmani_metadata=True,
                )
                matches = (
                    validation.h5_sha256 == shard.h5_sha256
                    and validation.json_sha256 == shard.json_sha256
                    and validation.h5_size_bytes == shard.h5_size_bytes
                    and validation.json_size_bytes == shard.json_size_bytes
                )
            except Exception:
                matches = False
            if not matches:
                if terminal or shard.episode_count == self.config.shard_size:
                    raise CollectionError(
                        f"sealed source shard failed resume validation: {h5_path}"
                    )
                manifest = self._repair_partial_shard(manifest, shard)
        persist_manifest(self.root, manifest)
        self._cleanup_committed_transactions(manifest)
        return manifest

    def _cleanup_committed_transactions(self, manifest: CollectionManifest) -> None:
        """Remove only journals whose authoritative manifest commit is durable."""
        inflight_root = self.root / "journal" / "inflight"
        if inflight_root.exists():
            for path in inflight_root.glob("candidate-*.json"):
                payload = _read_json_mapping(path)
                index = payload.get("candidate_scene_index")
                if isinstance(index, bool) or not isinstance(index, int):
                    raise CollectionError(f"malformed in-flight journal index: {path}")
                if index < manifest.next_candidate_scene_index:
                    path.unlink()
            if not any(inflight_root.iterdir()):
                inflight_root.rmdir()

        terminal = manifest.status is not CollectionStatus.IN_PROGRESS
        for shard in manifest.source_shards:
            if terminal or shard.episode_count == self.config.shard_size:
                directory = self._journal_shard_directory(shard.shard_index)
                if directory.exists():
                    shutil.rmtree(directory)
        journal_root = self.root / "journal"
        if journal_root.exists() and not any(journal_root.iterdir()):
            journal_root.rmdir()

    def _repair_partial_shard(
        self,
        manifest: CollectionManifest,
        shard: SourceShardRecord,
    ) -> CollectionManifest:
        groups = tuple(
            group
            for group in manifest.scene_groups
            if group.accepted
            and group.accepted_scene_group_id is not None
            and group.episodes
            and group.episodes[0].source_shard_id == shard.source_shard_id
        )
        bundle_paths = [
            self._group_bundle_path(shard.shard_index, group.accepted_scene_group_id or "")
            for group in groups
        ]
        if not groups or not all(path.is_file() for path in bundle_paths):
            raise CollectionError(
                f"partial shard cannot be repaired because journal bundles are missing: "
                f"{shard.source_shard_id}"
            )
        for group, bundle_path in zip(groups, bundle_paths, strict=True):
            self._require_trusted_group_bundle(
                bundle_path,
                group_id=group.accepted_scene_group_id or "",
                h5_sha256=group.bundle_h5_sha256,
                json_sha256=group.bundle_json_sha256,
            )
        build_directory = self.root / ".staging" / "resume-repair"
        build_directory.mkdir(parents=True, exist_ok=True)
        build_path = build_directory / f"shard-{shard.shard_index:05d}.h5"
        _remove_pair(build_path)
        build_source_shard(
            bundle_paths,
            build_path,
            shard_metadata={
                "collection_schema_version": manifest.collection_schema_version,
                "collection_run_id": manifest.collection_run_id,
                "source_shard_id": shard.source_shard_id,
                "shard_index": shard.shard_index,
                "expert_config_fingerprint": self.config.expert_config_fingerprint,
            },
        )
        canonical_h5 = self.root / shard.h5_path
        canonical_json = self.root / shard.json_path
        os.replace(build_path, canonical_h5)
        os.replace(build_path.with_suffix(".json"), canonical_json)
        validation = validate_native_archive_pair(
            canonical_h5,
            require_accepted_success=True,
            require_langmani_metadata=True,
        )
        if tuple(location.raw_trajectory_id for location in validation.episodes) != (
            shard.raw_trajectory_ids
        ):
            raise CollectionError("repaired partial shard raw trajectory order changed")
        locations = {
            location.raw_trajectory_id: location
            for location in validation.episodes
            if location.raw_trajectory_id is not None
        }
        repaired_records = {
            record.raw_trajectory_id: replace(
                record,
                native_episode_id=locations[record.raw_trajectory_id].native_episode_id,
                h5_group=locations[record.raw_trajectory_id].h5_group,
                h5_sha256=validation.h5_sha256,
                json_sha256=validation.json_sha256,
            )
            for record in manifest.raw_episodes
            if record.source_shard_id == shard.source_shard_id
        }
        raw_episodes = tuple(
            repaired_records.get(record.raw_trajectory_id, record)
            for record in manifest.raw_episodes
        )
        scene_groups = tuple(
            replace(
                group,
                episodes=tuple(
                    repaired_records.get(episode.raw_trajectory_id, episode)
                    for episode in group.episodes
                ),
            )
            if group in groups
            else group
            for group in manifest.scene_groups
        )
        repaired_shard = replace(
            shard,
            h5_sha256=validation.h5_sha256,
            json_sha256=validation.json_sha256,
            h5_size_bytes=validation.h5_size_bytes,
            json_size_bytes=validation.json_size_bytes,
        )
        return replace(
            manifest,
            raw_episodes=raw_episodes,
            scene_groups=scene_groups,
            source_shards=tuple(
                repaired_shard if item.source_shard_id == shard.source_shard_id else item
                for item in manifest.source_shards
            ),
        )

    def _journal_shard_directory(self, shard_index: int) -> Path:
        return self.root / "journal" / f"shard-{shard_index:05d}"

    def _require_trusted_group_bundle(
        self,
        bundle_path: Path,
        *,
        group_id: str,
        h5_sha256: str | None,
        json_sha256: str | None,
    ) -> None:
        bundle_json = bundle_path.with_suffix(".json")
        if (
            h5_sha256 is None
            or json_sha256 is None
            or not bundle_path.is_file()
            or not bundle_json.is_file()
            or sha256_file(bundle_path) != h5_sha256
            or sha256_file(bundle_json) != json_sha256
        ):
            raise CollectionError(
                "open shard refused an untrusted or changed group bundle: " + group_id
            )

    def _inflight_candidate_path(self, candidate_index: int) -> Path:
        return self.root / "journal" / "inflight" / f"candidate-{candidate_index:06d}.json"

    def _group_bundle_path(self, shard_index: int, group_id: str) -> Path:
        return self._journal_shard_directory(shard_index) / f"group-{_id_token(group_id)}.h5"

    def _source_shard_path(self, shard_index: int, shard_id: str) -> Path:
        return (
            self.root / "accepted" / "shards" / f"shard-{shard_index:05d}-{_id_token(shard_id)}.h5"
        )


def _require_location(draft: _AttemptDraft) -> NativeEpisodeLocation:
    if draft.native_location is None:
        raise CollectionError("selected attempt has no native trajectory location")
    return draft.native_location


def _require_raw_id(draft: _AttemptDraft) -> str:
    if draft.raw_trajectory_id is None:
        raise CollectionError("selected attempt has no stable raw trajectory ID")
    return draft.raw_trajectory_id


def _require_initial_scene_state(draft: _AttemptDraft) -> dict[str, Any]:
    if draft.initial_scene_state is None:
        raise CollectionError("selected attempt has no captured initial scene state")
    return json.loads(json.dumps(draft.initial_scene_state, allow_nan=False))


def _metadata_environment(recorder: Any) -> Any:
    """Return the project environment behind a recorder-like wrapper."""
    return getattr(recorder, "unwrapped", recorder)


def _normalize_initial_scene_state(value: object) -> dict[str, Any]:
    """Validate the explicit reset snapshot, including ManiSkill-static bins."""
    if not isinstance(value, Mapping) or set(value) != {
        "object_poses",
        "bin_poses",
        "panda_qpos",
    }:
        raise CollectionError(
            "initial scene state must contain object_poses, bin_poses, and panda_qpos"
        )

    def vectors(
        payload: object,
        *,
        expected_keys: tuple[str, ...],
        field_name: str,
        length: int,
    ) -> dict[str, list[float]]:
        if not isinstance(payload, Mapping) or set(payload) != set(expected_keys):
            raise CollectionError(f"{field_name} must contain exactly {', '.join(expected_keys)}")
        return {
            key: _finite_vector(payload[key], field_name=f"{field_name}.{key}", length=length)
            for key in expected_keys
        }

    return {
        "object_poses": vectors(
            value["object_poses"],
            expected_keys=("red_cube", "green_cube", "blue_cube"),
            field_name="object_poses",
            length=7,
        ),
        "bin_poses": vectors(
            value["bin_poses"],
            expected_keys=("left_bin", "right_bin"),
            field_name="bin_poses",
            length=7,
        ),
        "panda_qpos": _finite_vector(
            value["panda_qpos"],
            field_name="panda_qpos",
            length=9,
        ),
    }


def _finite_vector(value: object, *, field_name: str, length: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CollectionError(f"{field_name} must contain exactly {length} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CollectionError(f"{field_name} values must be real numbers")
        normalized = float(item)
        if not math.isfinite(normalized):
            raise CollectionError(f"{field_name} values must be finite")
        result.append(normalized)
    return result


def _materialize_evaluation(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(_EVALUATION_FIELDS):
        raise CollectionError("expert evaluation must contain exactly the eight M1 fields")
    result: dict[str, bool] = {}
    for key in _EVALUATION_FIELDS:
        field = value[key]
        if isinstance(field, bool):
            result[key] = field
            continue
        normalized = field
        if hasattr(normalized, "detach"):
            normalized = normalized.detach()
        if hasattr(normalized, "cpu"):
            normalized = normalized.cpu()
        if hasattr(normalized, "reshape") and hasattr(normalized, "tolist"):
            values = normalized.reshape(-1).tolist()
        else:
            raise CollectionError(f"expert evaluation {key} is not a scalar boolean")
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], bool):
            raise CollectionError(f"expert evaluation {key} is not a scalar boolean")
        result[key] = values[0]
    return result


def _expert_contract_failures(
    result: ExpertResult,
    *,
    scheduled: ScheduledEpisode,
    reset_episode_spec: EpisodeSpec | None,
    actual_evaluation: Mapping[str, bool] | None,
    attempt_index: int,
    config: CollectionConfig,
) -> list[str]:
    """Return every reason a successful M2 result is ineligible for replay."""
    if not result.success:
        return []
    reasons: list[str] = []
    expected_episode = EpisodeSpec.create(
        scene_seed=scheduled.scene_seed,
        task_spec=scheduled.task_spec,
    )
    if reset_episode_spec != expected_episode:
        reasons.append("reset EpisodeSpec differs from the deterministic schedule")
    if result.scene_seed != scheduled.scene_seed:
        reasons.append("expert scene_seed differs from the deterministic schedule")
    if result.scene_id != scheduled.scene_id:
        reasons.append("expert scene_id differs from the reset EpisodeSpec")
    if result.task_id != scheduled.task_id:
        reasons.append("expert task_id differs from the deterministic schedule")
    if result.canonical_instruction != scheduled.canonical_instruction:
        reasons.append("expert canonical instruction differs from the schedule")
    if result.target_object_id != scheduled.task_spec.target_object_id:
        reasons.append("expert target object differs from the schedule")
    if result.target_bin_id != scheduled.task_spec.target_bin_id:
        reasons.append("expert target bin differs from the schedule")

    recorded_evaluation = dict(result.final_environment_evaluation)
    if set(recorded_evaluation) != set(_EVALUATION_FIELDS):
        reasons.append("expert final evaluation does not contain the exact M1 fields")
    if actual_evaluation is None:
        reasons.append("fresh final environment evaluation is unavailable")
    else:
        for key in _EVALUATION_FIELDS:
            if recorded_evaluation.get(key) is not actual_evaluation.get(key):
                reasons.append(f"expert final evaluation disagrees with environment field {key}")
    required_values = {
        "target_in_target_bin": True,
        "target_in_wrong_bin": False,
        "wrong_object_in_target_bin": False,
        "target_is_grasped": False,
        "target_is_static": True,
        "target_off_table": False,
        "success": True,
        "fail": False,
    }
    for key, expected in required_values.items():
        if recorded_evaluation.get(key) is not expected:
            reasons.append(f"expert final evaluation {key} must be {expected!r}")
        if actual_evaluation is not None and actual_evaluation.get(key) is not expected:
            reasons.append(f"environment final evaluation {key} must be {expected!r}")

    expert_config = config.expert_config
    if result.total_environment_steps > expert_config.max_episode_steps:
        reasons.append("expert exceeded max_episode_steps")
    if attempt_index > config.maximum_expert_attempts_per_task:
        reasons.append("collector attempt exceeded maximum_expert_attempts_per_task")
    planning_bound = len(_PLANNED_PHASES) * expert_config.max_planning_attempts_per_phase
    if result.total_planning_calls > planning_bound:
        reasons.append("expert exceeded the aggregate planning-call bound")
    replan_bound = len(_PLANNED_PHASES) * max(
        0,
        expert_config.max_planning_attempts_per_phase - 1,
    )
    if result.total_replans > replan_bound:
        reasons.append("expert exceeded the aggregate replan bound")
    for phase_result in result.phase_results:
        if phase_result.phase in _PLANNED_PHASES:
            if (
                phase_result.attempts > expert_config.max_planning_attempts_per_phase
                or phase_result.planning_calls > expert_config.max_planning_attempts_per_phase
                or phase_result.replans > max(0, expert_config.max_planning_attempts_per_phase - 1)
            ):
                reasons.append(
                    f"expert phase {phase_result.phase.value} exceeded its planning-attempt bound"
                )
        elif phase_result.attempts > 1 or phase_result.planning_calls != 0:
            reasons.append(
                f"expert non-planning phase {phase_result.phase.value} exceeded its command bound"
            )
        if (
            phase_result.phase is ExpertPhase.SETTLE_AFTER_RELEASE
            and phase_result.environment_steps > expert_config.release_settling_steps
        ):
            reasons.append("expert exceeded release_settling_steps")
    return list(dict.fromkeys(reasons))


def _id_token(identifier: str) -> str:
    return identifier.rsplit("-", maxsplit=1)[-1][:20]


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {str(error) or repr(error)}"


def _remove_pair(h5_path: Path) -> None:
    h5_path.unlink(missing_ok=True)
    h5_path.with_suffix(".json").unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CollectionError(f"cannot read transaction journal {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise CollectionError(f"transaction journal root must be a mapping: {path}")
    return payload


def _episodes_by_id(json_path: Path) -> dict[int, Mapping[str, Any]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise CollectionError(f"source shard JSON is malformed: {json_path}")
    result: dict[int, Mapping[str, Any]] = {}
    for value in payload["episodes"]:
        if not isinstance(value, Mapping) or not isinstance(value.get("episode_id"), int):
            raise CollectionError(f"source shard episode metadata is malformed: {json_path}")
        result[int(value["episode_id"])] = value
    return result


def _attempt_for_raw(
    raw_id: str,
    new_attempts: tuple[AttemptRecord, ...],
    manifest: CollectionManifest,
) -> str:
    for attempt in (*new_attempts, *manifest.attempts):
        if attempt.raw_trajectory_id == raw_id:
            return attempt.attempt_id
    raise CollectionError(f"no attempt record owns raw trajectory {raw_id}")


__all__ = ["CollectionError", "RawDemonstrationCollector"]
