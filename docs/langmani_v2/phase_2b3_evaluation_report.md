# Phase 2B.3 evaluation report

## Decision

Phase 2B.3 closes as **Result B: engineering implementation valid, expert gate failed**. The
geometry/state-machine architecture is real and statically valid, but the remote regression
diagnostic produced a zero-tolerance wrong-object interaction and nine timeouts. The full
regression bank therefore stopped after 12 of 55 cases. Development was not authorized,
acceptance remained sealed, and the expert package is `REJECTED`.

The observed 2/12 diagnostic success count is not a development or independent success-rate
claim. Development executed 0/100 episodes, so its actual success rate is undefined rather than
0%. Acceptance also executed 0/100 episodes.

## Evidence lineage

- Branch base: `ee11c741f7537c8514f6a4c869ff276a9f757484`.
- Failure-evidence implementation: `e63931cb4e75908bf0447b6b1d46afee2ef34bf4`.
- Initial geometry expert: `d5e310e2a0a9e6584188c97e1537bb574d2e73f9`.
- Bounded budget-aware expert evaluated remotely:
  `3e7f805f7b9a0c0a230a61c27b83eb85d2e7c591`.
- Fail-fast evaluator/finalizer base: `dde96632800173cb34487959eaab77c0003d1240`.
- Evaluated expert configuration SHA-256:
  `17cb6df01ba33f10dfd5992650780c5a1697604243ae0c5c6d819b234c2e1c60`.
- Expert semantic digest:
  `9e0c1bd94f7745c35e711f35d19bfbcc27dbfd5e2789e76cdb7446641aef443a`.

## Historical audit and deterministic replay

The registry contains all 355 available Candidate E--P episode results: 184 successes and 171
failures. The largest failure groups are 107 stalled-progress cases, 25 unknown cases, 14
pre-approach planning failures, 8 object-workspace violations, 6 push-direction errors, 5 later
IK/motion-planning failures, 4 contact-loss failures, and 2 missed-contact failures.

Fifty-five selected failures were replayed twice. Every historical outcome was reproduced, repeat
semantics matched, and maximum repeated target-pose error was zero. Raw step traces and keyframes
remain outside Git; the repository contains only the compact registry, replay evidence, and root
cause summary.

The replay established three precise root causes:

- All five replayed workspace exits were the target object's footprint crossing a native boundary,
  not a robot TCP/link or waypoint-validator error. All were cylinders; signed margins ranged from
  -0.00907599 m to -0.00207788 m.
- All 13 replayed planning failures were deterministic native screw-planner rejections: eight at
  precontact, four during primary motion, and one at contact establishment. Nine were zero-step
  rejections. The shape split was ten cylinders and three boxes.
- All 13 replayed timeouts made progress before contact loss and a latched no-progress condition.
  Mean distance improvement was 0.250054 m; twelve were boxes and one was a cylinder.

## New expert architecture

`GeometryConstrainedPushExpert` is a new, versioned controller, not Candidate P with more branches.
It uses explicit `PLAN_PUSH`, `MOVE_TO_STAGING`, `MOVE_TO_PRECONTACT`, `ESTABLISH_CONTACT`,
`PUSH_CLOSED_LOOP`, `VERIFY_PROGRESS`, `RECOVER_CONTACT`, `REPLAN`, `SUCCESS`, and `SAFE_ABORT`
states. Each state has a frozen entry condition, target, progress rule, timeout, failure reason, and
legal successors.

Contact geometry uses direction-dependent box support and cylinder radius. Staging, precontact,
contact, TCP path, object path, and every bounded push segment are checked against a workspace
shrunk by support extent and controller margins before execution. Closed-loop verification reads
the current object pose, native contact feedback, target distance, lateral error, progress, and
cylinder angular speed. Recovery uses controller actions only, with at most two contact recoveries,
three global replans, twelve push segments, and 250 environment steps. It never teleports, mutates
an object pose or success flag, clips an invalid action, or applies a seed special case.

## Static and action gates

The final local gate covered 10,000 randomized geometry fixtures (5,000 boxes and 5,000 cylinders)
and 10,000 generated actions. Geometry errors, non-finite actions, shape errors, and action-bound
violations were all zero. Minimum object-workspace and TCP-workspace margins were 0.123416 m and
0.078905 m. The remote preflight also passed 10,000 plus 10,000 cases on the RTX 5090 runtime.

These checks validate the geometry generator and action contract. They do not substitute for a
successful simulator benchmark.

## Remote regression diagnostic

The first bounded smoke configuration completed 0/4 because eight short push segments exhausted
the available controller budget. The frozen budget-aware version retained the same architecture,
raised the bounded segment count to twelve, used at most 0.10 m segments, and reduced far-field
free-space execution cost. Its diagnostic result was:

| Metric | Observed |
| --- | ---: |
| Historical regression cases available | 55 |
| Diagnostic cases executed | 12 |
| Successes | 2 |
| Timeouts | 9 |
| Wrong-object interactions | 1 |
| Mean episode steps | 215.25 |
| Mean contact recoveries | 0.75 |
| Mean replans | 0.1667 |
| Workspace violations | 0 |
| Action-bound violations | 0 |
| Non-finite actions | 0 |
| Simulator errors | 0 |
| False successes | 0 |

Both successes were cylinders; the diagnostic cylinder split was 2/6 and the box split was 0/6.
Nine cases reached the 250-step horizon. Seed 66070 displaced a non-target object while moving to
staging and safely aborted after 25 steps. This is an observed new systematic safety failure and
triggers the regression fail-fast rule. The 43 remaining historical cases were not run, so the
evidence does not prove that all historical workspace or planning failures were eliminated.

## Development, acceptance, and package

- Development: `NOT_RUN_REGRESSION_GATE_FAILED`, 0/100 episodes, success rate unavailable.
- Acceptance: `NOT_RUN_DEVELOPMENT_GATE_FAILED`, sealed, 0/100 episodes.
- Expert package: `REJECTED`, collection entry point not authorized.
- Official data collection: 0 episodes and 0 frames.
- LeRobot exports, archives, SmolVLA batches, backward passes, optimizer steps, Phase 2C.2, and
  Phase 2D: not started.

The exact acceptance status intentionally follows the milestone contract even though development
was itself blocked by regression. No acceptance seed was loaded or revealed.

## Conclusion and pivot

The architecture improved the engineering boundary: geometry is explicit, workspace feasibility
is predictive, actions are valid by construction, control is closed loop, and recovery is bounded.
It did not meet the behavioral gate. Horizon exhaustion/contact loss remains dominant and a hard
box case introduced wrong-object displacement. These observations support Result B, not Result C:
the task has not been proven structurally impossible, but this one authorized rebuild failed.

The self-developed Push expert line should stop. The next authorized effort should use one of:
an official task with a reliable expert, an official benchmark dataset, validated motion-planning
demonstrations, or a stable existing LangMani task. Threshold relaxation and Candidate Q--Z are
not valid next steps.

## Reproducible commands

```powershell
$env:PYTHONPATH = "src"
python scripts/langmani_v2/build_push_failure_registry.py `
  --config configs/langmani_v2/phase_2b3/failure_audit.yaml
python scripts/langmani_v2/replay_push_expert_failure.py `
  --config configs/langmani_v2/phase_2b3/replay.yaml
python scripts/langmani_v2/validate_geometry_push_expert.py `
  --config configs/langmani_v2/phase_2b3/expert.json `
  --output artifacts/langmani_v2/phase_2b3/static_geometry_validation.json `
  --geometry-cases 10000 --action-cases 10000 --seed 0
python scripts/langmani_v2/evaluate_geometry_push_expert.py `
  --config configs/langmani_v2/phase_2b3/eval_regression.json `
  --output <run-root>/regression.json `
  --ledger <run-root>/regression.jsonl
python scripts/langmani_v2/finalize_phase2b3_rejection.py `
  --diagnostic-report <retrieved>/smoke12_3e7.json `
  --diagnostic-ledger <retrieved>/smoke12_ledger_3e7.jsonl `
  --remote-static-report <retrieved>/remote_static_d5e.json `
  --gpu-report <retrieved>/gpu.txt
```

The evaluator's no-limit command now fails fast on a zero-tolerance condition. The 12-case report
predates that final evaluator change, so the finalizer independently verifies the source SHA,
ledger equality, exact zero counters, wrong-object evidence, expert digest, and collection count
before creating the compact rejection artifacts.

## Remote execution audit

The remote runtime was Python 3.12.3, PyTorch 2.11.0+cu130, NumPy 1.26.4, ManiSkill 3.0.1,
SAPIEN 3.0.3, mplib 0.1.1, and an NVIDIA GeForce RTX 5090. Tracked source was clean for every
run. Run IDs, evaluated commits, runtime identity, and retrieved evidence hashes are recorded in
`artifacts/langmani_v2/phase_2b3/remote_execution_audit.json`. The server remains online per the
current user instruction.
