# Phase 2B.3 geometry-constrained closed-loop Push expert

## Objective

Replace the rejected Candidate E--P local-heuristic line with one separately versioned
`GeometryConstrainedPushExpert`. The new expert must use explicit geometry feasibility, a legal
state machine, closed-loop progress checks, and bounded contact recovery. It must reach at least
95/100 successes on a sealed acceptance bank with zero workspace, action-contract, nonfinite,
simulator, and false-success failures.

## Starting point

- Branch base: `ee11c741f7537c8514f6a4c869ff276a9f757484`.
- Candidate E remains the historical best formal baseline at 75/100; Candidate F remains 11/100.
- Candidate G--P evidence remains immutable diagnostic evidence. Candidate P remains 2/5.
- `langmani/phase2b-push-v2` has zero collected episodes and zero frames.
- Optimizer steps, SmolVLA training, Phase 2C.2, and Phase 2D remain zero/not started.

## Evidence-first sequence

1. Build a sanitized registry for every available Candidate E--P result. Missing historical fields
   stay explicitly unavailable rather than inferred.
2. Validate deterministic replay on all Candidate E failures, all five Candidate P probes,
   representative Candidate F failures, and at least five representatives of each sufficiently
   populated failure class.
3. Complete the root-cause summary and failure audit before changing expert control behavior.
4. Implement one new expert architecture, not Candidate Q or a Candidate P constant patch.
5. Pass 10,000-case static geometry and 10,000-action contract gates locally.
6. Commit and push the frozen implementation before any remote regression/development work.
7. Run the historical regression bank, then the 100-seed development bank.
8. Reveal and run the 100-seed acceptance bank once only if every development gate passes.
9. Package the expert only after acceptance; otherwise publish a rejected package and pivot.

## Seed partitions

- Historical/failure regression: only already consumed Candidate E--P seeds.
- Untouched Phase 2B.2 formal gate: 66300--66399; remains excluded.
- Untouched collection and top-up schedule: 67000--67809; remains excluded.
- Development bank: 69000--69099, repeatable and frozen.
- Acceptance bank: 69200--69299, frozen and inaccessible through the evaluation API until a
  content-bound passing development report is supplied.

The registry must prove pairwise disjointness and must reject a development command that asks for
acceptance seeds.

## Explicit exclusions

- No official data collection, LeRobot export, archive, dataset loader batch, optimizer step,
  SmolVLA work, Phase 2C.2, or Phase 2D.
- No teleport, direct object-pose mutation, success mutation, task-distribution narrowing,
  post-hoc action clipping, recorded-trajectory fallback, per-seed special case, or workspace
  expansion.
- No acceptance access before development passes and no adjustment after acceptance begins.
- No further Candidate Q--Z loop if this architecture misses the unchanged gate.

## Result contract

Result A requires every static/action/regression/development/acceptance gate and an accepted expert
package while official collection remains zero. Result B preserves an engineering-valid rejected
expert with exact development failures and untouched acceptance when development fails. Result C
requires reproducible evidence that the controller/task/workspace combination is structurally
infeasible. Any rejection recommends a task/data-source pivot instead of threshold relaxation.
