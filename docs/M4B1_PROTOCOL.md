# M4B.1 behavioral diagnostics (version 1)

Frozen before collecting diagnostic traces. M4A `717a07d` and M4B `99139de` remain unchanged.
This instrument measures observable behavior, not latent intent or language understanding.
The primary semantic selection measure is the first sustained destination approach, including
empty-handed motion. Contact and grasp of the shared red cube cannot distinguish left/right.

## Measurement definitions

Sample initial state and every post-action state at the existing 20 Hz control boundary.
Forces are the last physics substep, not every substep: brief contacts may be missed.
All positions are world meters. Left/right refer to semantic bins, not image coordinates.

* Intended goal: supplied prompt's semantic ID; null for blank. Requested goal remains separate.
* Object approach: TCP within 0.07 m of a cube center, with distance reduced by at least
  0.04 m from reset, for two consecutive samples. First qualifying cube; simultaneous
  first events are ambiguous, never broken by target identity.
* First approached goal / primary selected goal: TCP XY distance to a bin floor center
  at most 0.12 m, TCP height from floor between 0 and 0.20 m, XY distance reduced by
  at least 0.08 m from reset, for three consecutive samples (0.10 s between first and last
  sample at 20 Hz). First bin meeting
  this rule; an empty hand still reveals destination-directed motion. This is a proxy,
  not proof of intent, and is reported independently of successful placement.
* Object first contact: either Panda finger's pairwise resultant force with a cube has
  norm at least 0.1 N at one sampled boundary. Log all three objects. Hand/arm collisions
  are outside this signal. Destination first contact uses the same rule for either bin.
* First grasp object: installed ManiSkill Panda `is_grasping` defaults (both fingers
  at least 0.5 N and force/opening angle at most 85 degrees), for two samples. Grasp
  goal side is unavailable because the red cube is common to both goals.
* Placement attempt goal: after a red grasp and red lift at least 0.02 m above its
  reset height, red cube center lies within both bin XY half widths (0.085 m), height
  from floor in [0, 0.15] m, and descends at least 0.02 m from its post-grasp peak.
  Require two samples; release/static/full containment are not required.
* Final goal: red center inside a bin's XY half widths (0.085 m), height above floor
  in [0, 0.15] m at final sample. This relaxed geometric correspondence is separate
  from unchanged M1 full success. A red cube still on the source table has no final goal.
* Grasp then drop: after confirmed grasp and lift, red is ungrasped for three samples,
  outside both relaxed bin volumes. This describes a release outside a destination,
  not its cause; a later recovery does not erase the event.
* Stall: final 20 TCP samples have maximum distance from their first point <=0.01 m.
  Oscillation: last 40 TCP samples travel >=0.10 m with endpoint distance <=0.015 m.

All thresholds are measurement conventions fixed before results. No thresholds are fitted to
success labels. Null and ambiguous events count as non-selection, not wrong-side selection.
Initial samples cannot create approach, contact, grasp or placement events.

## Scoring and taxonomy

Each of the original 100 physical rollouts retains its identity; canonical trajectories are
reused for correct/swapped scoring and each blank for two requested goals: 160 scored rows.
Report requested-goal and supplied-goal metrics separately, including swapped behavior.
Blank has no supplied semantic goal: supplied-goal denominator is zero, value null.
For each scoring reference, report all-row counts/denominators, and completion conditional
on first correct destination selection. Object-contact/grasp accuracies explicitly measure
red-object selection only. The side-discriminating contact metric is bin contact, which is
not necessary for successful manipulation. Report a cumulative task funnel (object approach,
red contact, red grasp, correct destination approach, correct placement attempt, success)
as well as non-cumulative stages; funnel is achievement-based, not assumed event order.

First-match failure precedence (success is separate): infrastructure_error; wrong_goal_selected
(non-null first destination differs); correct_goal_wrong_destination (first approach correct,
later placement/final geometry wrong); correct_goal_no_contact (no red contact ever);
correct_goal_contact_no_grasp (no red grasp ever); correct_goal_grasp_then_drop;
correct_goal_execution_timeout; other (selected correct with another termination).
For absent destination selection: oscillation_or_stall; no_meaningful_interaction (no
object approach/contact/grasp); object_approached_no_contact; object_contact_no_grasp;
object_grasp_no_destination; other. These last categories do not diagnose misunderstanding.
Blank physical timeout taxonomy uses `unprompted_destination_*` categories when a side
was selected; no invented intended goal is assigned. Preserve requested-goal scores separately.

Paired categories, with precedence: both_corresponding; same_target (both non-null and
equal); only_one_corresponding; indeterminate (either absent); both_wrong_opposite.
The categories partition all 20 scene pairs. `paired_instruction_switch_accuracy` is
both_corresponding / 20 from first approaches, separate from both full successes / 20.
Also retain overlapping raw one-correct count so same-target pairs do not hide it.

## Validation and provenance

Replaying recorded applied actions avoids substituting a different run for the original 85
timeouts. Verify immutable source file/action/video hashes, all reset state/RGB/qpos hashes,
all final original flags, termination and lengths. Re-encode videos with the identical M4B
encoder and require exact video hashes; this binds the complete rendered sequence.
Compare instrumented/uninstrumented per-step state hashes on deterministic representatives
of each of the five physical prompt IDs. Re-evaluate a frozen checkpoint with an observational
wrapper around unchanged M4B `run_policy`, requiring original action and video equality.
Every sensor sample asserts before/after public state digest equality. No instrumentation
enters model observations, control or success scoring. Tests cover boundaries, ties, temporal
persistence, swapped/null semantics, taxonomy precedence and pair partitions.

Representatives use ascending (numeric scene seed, prompt ID), first qualifying example in
each category; absent categories explicitly say unavailable. Include both trajectories of
the first successful switching pair. New evidence has fresh paths; originals are read-only.
