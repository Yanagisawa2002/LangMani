# Phase 2B.5: Official ManiSkill Multi-Skill Demonstration Source Qualification

## Scope

Phase 2B.5 asks whether pinned official ManiSkill demonstrations can form a reproducible,
replay-valid, multi-skill source under one future-student contract. It may download and hash a
bounded shortlist, replay recorded actions, reconstruct a five-episode-per-task visual/state
pilot, and round-trip only that pilot through LeRobot 0.6.

It does not create a planner or expert, repair official actions, change official success
predicates, export a full dataset, create an optimizer, run a backward pass, or train a policy.

## Completion

Phase 2B.5 completed as `RESULT_A` at execution commit
`0b7f4d5e3b6fc909ee370aad5048ea7cc3ab9b17`. Three direct official sources passed every frozen
gate: `PickCube-v1`, `StackCube-v1`, and `PushCube-v1`. The compact 28-file artifact package and
independent no-simulator verifier both pass. This result validates only the official source and
makes a separately invoked Phase 2B.6 dataset-production phase eligible; no full dataset or policy
training was started.

## Immutable start and isolation

- Parent: `codex/langmani-v2-phase2b4-f1-ppo-diagnosis` at
  `6ca3786702ada4e92be607a5293b9c3c31d2a7d4`.
- Branch: `codex/langmani-v2-phase2b5-official-demos`.
- Worktree: `D:\LangMani-worktrees\phase2b5-official-demos`.
- Official source: `haosulab/ManiSkill_Demonstrations` at immutable revision
  `d674485bbffdd533914e52d272fdda34c0515608`.
- Official source bytes remain outside the repository. Generated camera and LeRobot pilots remain
  outside source control.

## Ordered gates

1. Verify the parent, clean state, remote identity, F1 evidence, no relevant jobs, and zero mixing
   with custom dataset identities.
2. Freeze the custom push route and all downstream training authorizations.
3. Verify official download/replay mechanisms, release identity, dataset revision, and license.
4. Inspect all five required candidate archives and actual HDF5/JSON data.
5. Accept only direct `Panda / pd_joint_pos / float32[8] / 20 Hz` sources; do not clip, project, or
   convert an incompatible source.
6. Replay 20 stratified episodes per candidate direct source using only recorded actions and the
   official first-state anchor.
7. For every passing selected task, replay 100 stratified episodes.
8. Reconstruct five passing visual/state episodes per selected task with base-camera
   `uint8[256,256,3]`, `PandaPolicyStateV0 float32[9]`, recorded actions, time, task identity, and
   one canonical instruction.
9. Convert and read back only those 15 pilot episodes in an isolated LeRobot 0.6 environment.
10. Audit future padding and design episode-level, reset-aware, task-aware, language-group-aware
    splits without materializing them.
11. Classify exactly Result A/B/C/D and stop without training.

## Frozen replay and acceptance gates

- Categorical outcome agreement: at least 99%.
- Action/frame alignment: exactly 100%.
- Invalid action count: zero.
- Simulator error count: zero.
- Selected source diversity: at least three genuine skill families.
- Student fields: RGB, Panda state, action, instruction, task ID, and timestamp only.
- LeRobot pilot: real public-API creation, decode, metadata, random access, shape, task, episode,
  frame, and timestamp readback.

Result A grants only `phase2b6_dataset_production_eligible=true`. Full production and ACT, SmolVLA,
VLA-JEPA, or any other policy training still require a separate instruction.
