# Phase 2B.3 historical Push failure audit

## Scope and evidence boundary

This audit is the evidence-first entry point for the single authorized Phase
2B.3 architecture rebuild. It consumes only the compact, content-bound reports
from Candidates E through P. It does not run a replacement controller, collect
training data, touch official collection seeds, or reinterpret a heuristic as
simulator evidence. The exact input identities and report SHA-256 digests are
frozen in `configs/langmani_v2/phase_2b3/failure_audit.yaml`.

The registry contains 355 historical diagnostic episodes: 184 successes and
171 failures. Its records digest is
`sha256:6553d4c2202c6d4a82bc24c3fcc53e1ad19d4d79d0cde075f063e43a36b761f1`.
The official collection episode count remains zero.

## Failure taxonomy

| Failure category | Count |
| --- | ---: |
| Stalled progress | 107 |
| Unknown from compact historical evidence | 25 |
| Pre-approach planning failure | 14 |
| Object workspace violation | 8 |
| Push-direction error | 6 |
| IK or later motion-planning failure | 5 |
| Loss of contact | 4 |
| Missed contact | 2 |

The last completed phase was `primary_push` for 87 failures,
`corrective_push_1` for 35, `move_to_precontact` for 27, `verify_task` for 13,
and one of the remaining later phases for 9. Boxes account for 83 failures and
cylinders for 88. All observed loss-of-contact, push-direction, and object
workspace violations occurred on cylinders in the compact historical reports.

## What the compact evidence establishes

- A timeout paired with `no_progress_stall` is classified as stalled progress;
  timeout alone is not treated as a root cause.
- Historical phase evidence separates pre-approach failures from failures after
  approach selection, but it cannot by itself distinguish IK infeasibility from
  a later target-pose or motion-planning failure.
- The native environment's workspace flag refers to the target object's full
  planar footprint. The minimum reported target-object workspace margin is
  `-0.0020981621742248505`. The compact reports contain no robot-link workspace
  violation flag, so they do not support attributing those eight failures to a
  planned robot waypoint.
- Only two historical failures contain both a phase boundary and a detailed
  diagnostic trace. The other 169 failures provide phase-boundary evidence
  only. Detailed contact-loss, drift, approach-feasibility, and post-contact
  attribution therefore requires deterministic replay.

## Deterministic replay gate

`scripts/langmani_v2/replay_push_expert_failure.py` checks out the exact source
commit for each selected historical case and runs two fresh simulator episodes
per case. It records per-step object poses, target pose, robot state, TCP pose,
velocities, action-bound margin, footprint-aware workspace margin, a contact
proxy, evaluation state, action digest, and initial/final RGB keyframes. Raw
traces and images remain outside Git; only a compact validation report may be
retrieved.

The replay set includes every Candidate E failure, every Candidate P result,
stratified Candidate F failures, and enough additional historical failures to
cover each available category at least five times where the archive permits.
Each case must reproduce the historical outcome and step count, repeat the same
semantic result twice, and keep maximum target-pose repeat error at or below
`1e-5`.

No new expert behavior is authorized until this replay gate has run on the
pinned native Linux GPU stack and the resulting evidence has been reviewed.

## Seed isolation

Development seeds are frozen at `69000..69099`. Acceptance seeds are frozen at
`69200..69299` and remain sealed until a content-bound 100-episode development
report records at least 95 successes and zero action-contract, simulator,
workspace, teleport/state-mutation, success-flag-mutation, and execution
failures. All historical, reserved Phase 2B.2 formal, official collection, and
top-up ranges are explicitly excluded by the seed registry.
