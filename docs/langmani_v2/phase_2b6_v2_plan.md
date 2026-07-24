# LangMani 2.0 Phase 2B.6-v2 plan

Phase 2B.6-v2 is a new, non-resumable production identity for the three pinned
official ManiSkill demonstration sources. It starts from source episode 0 and
does not reuse any of the 1,938 Phase 2B.6 partial replay records.

## Frozen objective

Produce a nonprivileged, LeRobot 0.6-compatible multi-skill dataset while
retaining every one of the 3,000 source identities. A source episode may be
absent from policy data only through the separately versioned exclusion policy.
No policy model, checkpoint, optimizer, backward pass, planner, or expert is in
scope.

## Source and output identities

- Source commit:
  `18c31f8bc6839d80b465ad73894fa886fa6c8dfb`
- Producer branch:
  `codex/langmani-v2-phase2b6-v2-instrumented-production`
- Dataset identity: `LangManiOfficialMultiSkill-v2`
- Production root:
  `/root/autodl-tmp/langmani-external/phase2b6-v2/production_v2`
- Archive root:
  `/root/autodl-tmp/langmani-external/phase2b6-v2-archive`
- Restore scratch:
  `/root/autodl-tmp/langmani-external/phase2b6-v2-restore-scratch`

The Phase 2B.6 v1 partial root remains read-only.

## Ordered gates

1. Rehash Phase 2B.5, 2B.6, 2B.6.1, 2B.6.1-R, and 2B.6.1-v2 evidence.
2. Rehash all official ZIP and selected JSON/HDF5 source bytes.
3. Freeze the exclusion policy and complete production specification.
4. Pass the recovered Vulkan/SAPIEN launcher, minimal SAPIEN probe, and
   zero-step construction for all three tasks.
5. Pass the three-task physical-equivalence visual-shift pilot.
6. Replay PickCube, StackCube, and PushCube in source episode order.
7. Persist every source classification and all explicit replay sub-gates.
8. Materialize only accepted episodes into LeRobot task/split roots.
9. Recompute accepted-only folds, padding, statistics, normalization, and task
   balance.
10. Read every accepted episode and decode every generated video frame.
11. Independently verify source-to-derived equality and excluded-source
    absence.
12. Audit leakage, create the content-addressed archive, and restore into clean
    non-symlink scratch bytes.
13. Create the accepted package only when every gate passes.

## Hard stops

Production stops on changed evidence or source bytes, Vulkan failure, missing
action identity, any unclassified failure, more than five exclusions for one
task, more than nine exclusions overall, accepted-data alignment or readback
failure, privileged-data leakage, primary-split overlap, or archive/restore
failure. Physical-quality failures are never retried. Infrastructure failures
retain the first attempt and permit at most one identical retry.

## Authorization boundary

A Result A may set ACT-baseline, SmolVLA, and VLA-JEPA dataset eligibility to
true. It cannot authorize or start training. Throughout this phase,
`student_policy_training_started=false`, `optimizer_created=false`,
`backward_passes=0`, and `optimizer_steps=0`.
