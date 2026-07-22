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

Candidate H completed 64/64 diagnostic episodes but reached only 38 successes. Full-resolution
precontact execution averaged 104.9 steps and caused fifteen timeouts, primarily in the first
correction. Candidate I retains candidate-specific retry but samples only the initial vertical lift
and final descent at the already content-bound free-space stride; the obstacle-sensitive high
staging motion remains full resolution. When containment is first observed, the controller now
holds its current joint position for the existing bounded settle window so stable success can form
before any endpoint overshoot or later verification. Diagnostic probes also stop and write evidence
as soon as a zero-tolerance failure occurs or their mathematical success ceiling falls below 95%;
formal gates never early-stop under this rule.

Candidate I passed full local implementation validation at Git
`3da6ecf5afccc09eae5fe69ece9170647a3357a0`. Its diagnostic-only range is 66500--66563. The
untouched formal range remains 66300--66399, and all collection seeds remain disjoint. The v2
contract binds `PushToRegionExpert/CandidateI` to the implementation commit without changing the
formal threshold or zero-tolerance criteria.

Candidate I stopped honestly at 12/16 once its best possible 64-episode result fell to 60/64.
Candidate J addresses only the four recorded failure mechanisms: zero-step lift fallback, bounded
stabilization with correction after containment drift, stride-two contact establishment with exact
terminal targets, and six 8 cm fallback segments after a failed direct plan. Cylinder pushes use
the same 3.5 cm interior margin as the cube. Formal seeds and collection remain sealed.

Candidate J passed local implementation validation at Git
`d757d91fa7c12f8c341f692118d8d45882623834`. Its diagnostic-only schedule is 66600--66663. The
formal 66300--66399 range is still untouched, and the v2 contract binds the candidate source without
changing any acceptance threshold.

Candidate J stopped after 14 diagnostic episodes because a target-workspace exit triggered the
zero-tolerance rule. Its ten successes and three other failures are retained as rejected evidence.
Candidate K removes stride sampling from both contact motions, adds a deterministic intermediate
contact waypoint after a zero-action screw-plan failure, sets the fixed cylinder goal margin to
2 cm, and uses eight 6 cm maximum fallback segments. Diagnostic seeds 66700--66763 remain isolated;
formal and collection schedules remain sealed.

Candidate K passed focused runtime-contract and Ruff validation at implementation commit
`413710deb15a1139b3ac830387c7543d7839c7f7`. The contract binds that exact commit without changing
the 95/100 formal threshold or any zero-tolerance criterion.

Candidate K stopped after five diagnostic episodes because its first cylinder/left task left the
target workspace. Candidate L is limited to object-specific pre-containment braking while preserving
the same planner targets, controller, success geometry, horizon, and formal gate. Its diagnostic
schedule will remain disjoint from every earlier probe and from formal collection.

Candidate K stopped after 5/64 diagnostic episodes on a zero-tolerance cylinder workspace exit;
the other failure was a correction timeout. Candidate L retains the fixed paths but brakes a push
when privileged target distance enters a fixed band just outside full containment, then holds the
current measured controller state for the existing settle window. This prevents primary-push
momentum from continuing while the arm repositions. Diagnostic seeds 66800--66863 remain isolated.

Candidate L is content-bound to implementation commit
`c7361c60daeaa50740bccee213de9485a4c078ef`; its diagnostic schedule cannot promote the expert.

Candidate L stopped after four standard timeouts because its best possible full-probe result fell
to 60/64 (93.75%). With the two complete formal gates at 75/100 and 11/100 and all G--L probes
rejected, Phase 2B.2 closes as Result B at the mandatory expert gate. Formal seeds 66300--66399,
collection seeds, export, archives, SmolVLA loading, training, and Phase 2D remain untouched.

The original task authorization also requires expert repair before declaring the data path terminal.
Candidate M therefore resumes diagnostic screening after the Result B snapshot: it preserves the
safe precontainment brake and replaces expensive high re-contact with one bounded 8--25 mm
in-contact correction. Seeds 66900--66963 are diagnostic only and formal collection stays closed.

Candidate N preserves the successful cube behavior and uses an earlier 5 cm cylinder braking band
with at most 15 mm per in-contact nudge. Its diagnostic seeds 68000--68063 are disjoint from the
formal gate and the reserved 67000-series collection/top-up schedule.
