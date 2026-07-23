# Phase 2B.6 Dataset Card

## Dataset status

This is a card for a rejected production candidate, not an accepted dataset release.
`accepted_multiskill_dataset_validated=false`.

1. **Official sources.** The verified source catalog contains PickCube-v1, StackCube-v1, and
   PushCube-v1 from `haosulab/ManiSkill_Demonstrations` revision
   `d674485bbffdd533914e52d272fdda34c0515608`, licensed Apache-2.0.
2. **Skills.** The intended skills are pick and place, stacking, and planar pushing. The partial
   derived output completed PickCube, partially produced StackCube, and never started PushCube.
3. **Episodes and frames.** Source authority has 3,000 episodes and 254,374 transitions. The
   rejected production attempted 1,939 episodes, accepted 1,938 replay records with 178,633 frames,
   rejected one, and left 1,061 unattempted. There is no final derived total.
4. **Observation and action.** Intended student observations are uint8 RGB
   `observation.images.base_camera[256,256,3]` plus `PandaPolicyStateV0 float32[9]`.
   Actions are native `pd_joint_pos float32[8]` at 20 Hz with no clipping, projection,
   interpolation, or fabricated terminal action.
5. **Language.** Each task has six train, two validation, and four held-out reviewed templates.
   One template identity is deterministically bound to each candidate episode.
6. **Splits.** Candidate assignments are 700/100/100/50/50 per task for train, validation,
   unseen reset, unseen task language, and visual shift. They are source-level assignments, not an
   accepted dataset split release.
7. **Unseen reset.** It holds out distinct task-local episode/reset identities. The source exposes
   no broader reset-family label, so it cannot prove family-level reset generalization.
8. **Unseen language.** It uses held-out template families absent from the train bank. No model
   evaluation was run.
9. **Cross-skill folds.** The three leave-one-skill-out folds are metadata references only. They
   require separate future models and currently are not eligible because production failed.
10. **Visual shift.** The registered transform multiplies post-render exposure by 0.9 and RGB
    channels by `[1.03, 1.0, 0.94]`. Its 3/3 pilot passed; the 150-episode shifted set was not built.
11. **Padding.** Horizons 10, 16, and 50 use an explicit boolean `action_is_pad`; masked losses
    must exclude padded timesteps.
12. **Normalization.** No canonical normalization exists. Train-only statistics were not run
    after the hard stop.
13. **Task imbalance.** Candidate train episodes are balanced 700 per task, but frame frequency
    favors StackCube. Uniform-task sampling was recommended for a future accepted dataset only.
14. **Privileged fields.** The frozen writer schema excludes object pose, goal pose, simulator
    state, reward, success internals, expert phase, and future observations. Full LeRobot readback
    did not run.
15. **Replay and readback.** No. PickCube replay completed, StackCube stopped at episode 938, and
    PushCube did not start. Full LeRobot conversion and readback did not start.
16. **Storage.** Partial bytes are at
    `/root/autodl-tmp/langmani-external/phase2b6/production_v1`. No accepted primary or archive copy
    exists.
17. **Restoration.** No clean restore was attempted because no accepted archive was created.
18. **Eligibility.** ACT, SmolVLA, and VLA-JEPA experiment eligibility are all false.
19. **Authorization.** All training remains unauthorized because source-to-derived production
    integrity failed before a complete, readable, leakage-audited, archived, and restored package
    existed.

## Known failure

`StackCube-v1` source episode 938 was a non-retryable post-execution replay-gate rejection. The
original wrapper failed to persist the returned inner subgate record, so the exact inner failure
is unavailable and was not inferred. This diagnostic limitation is part of the immutable Result C
evidence.
