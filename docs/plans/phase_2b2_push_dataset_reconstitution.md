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

Candidate F then completed the frozen 66100--66199 gate at source
`f441b1b035007f0876dafd2a54886b26f6350e40` and was rejected at 11/100. Its report SHA-256 is
`94975a838063c4e20891cf478bf775127eaf6a04fbcfcd990caef415fad3087d`. Formal collection remains
closed. Any successor must reduce controller-executed primary-push horizon cost while preserving
the 250-step limit, exact success criterion, obstacle-aware approach constraints, native action
bounds, and all zero-tolerance safety gates. A successor requires another untouched 100-episode
schedule and a new versioned identity.

Candidate G is implemented at Git `24709f1702914e72131245484d4405ac68a6303c`. It restores one
direct content-bound primary-push plan as the default and uses the bounded segmented path only when
that direct plan fails before executing an action. It also records complete fallback step counts,
raises obstacle-crossing staging without changing the physical contact height, permits the existing
controller to perform lateral-cylinder correction, and uses a geometry-specific cylinder endpoint
margin. A diagnostic-only 32-episode probe may use seeds 66200--66231; probe output is explicitly
ineligible for formal promotion. The untouched formal gate is frozen at seeds 66300--66399 under
`PushToRegionExpert/CandidateG`. The threshold, 250-step horizon, task predicate, native action
bounds, and every zero-tolerance gate remain unchanged.

The Candidate G diagnostic probe completed 32/32 but reached only 22 successes. The ten failures
comprised four planning failures, two timeouts, two wrong-object interactions, one correction
failure, and one target workspace exit. Six failures were in `move_to_precontact`; four were in the
corrective phases. Candidate G is rejected without touching 66300--66399. Candidate H therefore
splits high staging from candidate-specific descent so a zero-action deterministic screw-planning
failure may try the next predeclared safe candidate, executes approach plans at full controller
resolution, and stops a primary or corrective motion as soon as full containment is observed so
settling occurs without continuing toward an overshooting endpoint. It does not repair actions,
mutate state, change task success, or extend the horizon.

Candidate H passed local implementation validation at Git
`62db322a6954a362078badcb5f298501cb533bcf`. A 64-episode diagnostic-only probe is reserved at
66400--66463, disjoint from the untouched 66300--66399 formal gate and from all collection ranges.
Probe evidence cannot promote the expert. It only determines whether spending the formal schedule
is justified. The v2 contract now binds `PushToRegionExpert/CandidateH` to the implementation
commit while leaving every acceptance threshold unchanged.
