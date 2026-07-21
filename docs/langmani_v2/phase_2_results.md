# LangMani 2.0 Phase 2 results

## Recruiter summary

LangMani 2.0 added a genuinely different manipulation family: a Panda must push either a cube or a
horizontal rolling cylinder into one of four planar regions without grasping, lifting, toppling, or
moving a distractor. The environment, task taxonomy, vectorized evaluation, deterministic reset,
privileged mplib expert, diagnostics, and unified evaluator compatibility are implemented.

The initial fixed expert gate failed at 41/50 standard and 21/30 hard. Phase 2A then diagnosed the
side-push failures at expert phase boundaries and tested a bounded sequence of general controller
changes. The accepted implementation improved the unchanged fixed protocol to 46/50 standard and
23/30 hard, while preserving all 24 standard forward successes and eliminating workspace-exit and
action-bound events. The expert gate now passes without filtering seeds or weakening success.

## Accepted native results

The accepted producer is Git `59ca88e9f0514187252a6286ab1b8e06c4318fb4`. It ran on one RTX
5090 with the pinned planner environment `numpy==1.26.4` and `mplib==0.1.1`.

| Evaluation | Standard | Hard | Gate |
| --- | ---: | ---: | --- |
| Fixed lateral subset | 22/26 | 11/16 | diagnostic only |
| Exact 16-task smoke | 8/8 | 6/8 | passed |
| Fixed 50/30 target validation | 46/50 (92%) | 23/30 (76.7%) | passed |

The protected standard forward-left/forward-right subset remained 24/24. Every report completed its
exact schedule in order, with no duplicate episode identities and no command errors. Wilson 95%
intervals for the target validation are 81.2%-96.8% for standard and 59.1%-88.2% for hard.

Target-validation status counts:

| Difficulty | Success | Verification | Wrong object | Workspace exit | Action bounds | Timeout | Planning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 46 | 4 | 0 | 0 | 0 | 0 | 0 |
| Hard | 23 | 3 | 4 | 0 | 0 | 0 | 0 |

The four remaining standard failures are cylinder verification failures at seeds 44012, 44028,
44029, and 44044. Hard has cylinder verification failures at 45013, 45020, and 45029, plus
wrong-object precontact interactions at 45000, 45002, 45018, and 45024. These are valid
controller-quality outcomes, not infrastructure errors.

## What changed

- Added phase-boundary side-push traces and deterministic root-cause classification.
- Added a small scored lateral approach set using alignment, reachability, workspace, and distractor
  clearance, with separate cube and horizontal-cylinder geometry.
- Applied the measured 15-degree signed compensation only to lateral cube pushes.
- Disabled the unsafe lateral-cylinder re-contact sweep; incomplete primary pushes still fail the
  unchanged final verification.
- Added an expert-only action-bound accessor and rejected a complete joint plan before any invalid
  action could be stepped. A zero-step bounds rejection may try only the next already-scored lateral
  approach; no action is clipped or projected.
- Preserved forward target positions, forward control flow, success predicates, target regions, the
  250-step budget, and fixed schedules.

Candidate A caused six cylinder workspace exits. Candidate B improved lateral success but retained
three standard workspace exits. Candidate C bounded correction endpoints, but traces proved the
cylinder was swept during re-contact before that bound took effect. Candidate D removed the unsafe
re-contact and restored zero workspace exits; Candidate E then removed the remaining action-bound
event at its originating layer and passed both official gates. Two still earlier uncommitted probes
remained rejected at 10/17.

## What exists and what does not

Implemented and physically exercised:

- `LangMani-PushToRegion-v0` with 16 semantic task instances;
- conservative success/failure metrics and no-leakage observations;
- deterministic Panda/mplib pushing expert with explicit phases;
- behavior-neutral failure replay, full lateral audit, exact smoke, and fixed 50/30 validation;
- preserved machine-readable results with seed-level failure attribution.

Not started in Phase 2A:

- pushing demonstration collection;
- multi-skill LeRobot dataset and split manifests;
- SmolVLA dependency/runtime adapter;
- SmolVLA smoke or full training;
- learned push-only or multi-skill closed-loop evaluation.

## Exact next step

The expert quality gate no longer blocks demonstration collection. The next separately authorized
stage may collect and independently validate a deterministic pushing archive under the frozen task,
seed, action, and success contracts. SmolVLA must remain downstream of accepted data and does not
start automatically from this result.
