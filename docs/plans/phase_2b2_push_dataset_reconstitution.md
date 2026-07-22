# Phase 2B.2 push dataset reconstitution plan

## Objective

Create a newly collected and independently identified `langmani/phase2b-push-v2` dataset without
recovering or impersonating v1. Keep the native ManiSkill episodes as raw authority, action-replay
every generation-accepted trajectory in a fresh environment, export only replay-passing episodes
through LeRobot 0.6.0, then content-address and restore-test the data before any future training.

## Ordered gates

1. Freeze the v2 contract and retire v1 from runtime use.
2. Pass a disjoint 100-episode expert gate at at least 95% success with zero simulator, finite-action,
   action-bound, and workspace violations.
3. Collect a resumable 650-attempt initial schedule and, only after replay, one bounded 160-attempt
   top-up if the 500 preferred target or minimum strata remain deficient.
4. Fresh-environment action replay of every generation-accepted episode.
5. Export at least 400 accepted successes and 50,000 frames with real LeRobot metadata/readback.
6. Run full quality, integrity, duplicate, split-leakage, and native/archive identity checks.
7. Build content-addressed raw and LeRobot archives and restore-test them.
8. Verify replication and one real SmolVLA loader batch with zero optimizer steps.

## Assumptions

- The accepted Candidate E implementation remains an ancestor of the clean collection commit, but
  the stricter v2 expert gate may require a separately versioned successor before collection.
- The target runtime remains NumPy 1.26.4, mplib 0.1.1, ManiSkill 3.0.1, LeRobot 0.6.0, and one
  native NVIDIA GPU.
- Six historical semantic splits are retained because they are stricter than a single 80/10/10
  random split and preserve the existing Phase 2C development/external-evaluation boundary.

## Exclusions

- No v1 byte reconstruction or identity reuse.
- No failed episode enters the default imitation-learning view.
- No metadata editing after export.
- No SmolVLA backward pass, optimizer step, tiny overfit, formal training, checkpoint work,
  closed-loop evaluation, final access, Phase 2C.2, or Phase 2D.
- Raw episodes, videos, LeRobot data, archives, caches, and full runtime logs remain outside Git.

## Expert-gate amendment

The first exact v2 gate at source `cd97c4f7cb48be5d45569a5729926fe7ddaa3075` completed all
100 episodes but was rejected at 75/100. The immutable report digest is
`sha256:a6a8296a9df7c716698920d782c8e6f0aead6f61f18c228dcadba0a13a1c9489`; it records seven
planning failures, seven timeouts, seven verification failures, four wrong-object interactions,
zero simulator errors, zero non-finite actions, zero action-bound violations, zero workspace
violations, and zero optimizer steps. Formal collection remains closed.

The bounded repair is a new expert identity rather than a relaxation of the gate. It adds a fixed
high staging pose and obstacle-aware approach candidates for every direction, preserves the exact
final free-space planner target while sampling only intermediate free-space actions, and replaces
the long primary push with at most twelve state-aware 4 cm segments with one deterministic smaller
planning fallback. The environment, task geometry, success predicate, controller, 250-step limit,
action-bound rejection, seeds, and 95% threshold remain unchanged. The rejected gate is diagnostic
evidence only; a successful promotion must use a new untouched 100-episode schedule.

Local unit, repository, lint, format, and changed-module type validation passed for Candidate F at
Git `6277d77227a51d39d675e328c4df7c233f302484`. The v2 contract now binds that implementation as
`PushToRegionExpert/CandidateF` and freezes the untouched expert-gate seed interval 66100 through
66199. The v1 collection contract remains bound to Candidate E; neither its dataset identity nor
historical episode provenance is rewritten. Candidate F may open collection only if this newly
frozen gate reaches the unchanged 95% threshold.
