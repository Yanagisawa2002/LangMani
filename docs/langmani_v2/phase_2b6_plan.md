# Phase 2B.6: Official Multi-Skill Dataset Production

## Scope

Phase 2B.6 converts the accepted Phase 2B.5 official PickCube, StackCube, and PushCube sources into
the first complete nonprivileged LangMani multi-skill package. It replays recorded actions,
captures pre-action RGB and Panda state, materializes leakage-resistant episode splits and
language, writes public LeRobot 0.6 roots, computes explicit padding and train-only
normalization, builds a multi-root index, archives the package outside Git, and validates a clean
restore.

It does not load or train ACT, SmolVLA, VLA-JEPA, Diffusion Policy, SARM, PPO, or any other policy.
It does not create an optimizer or execute a backward pass. The custom push expert route remains
closed.

## Immutable start and locations

- Source branch: `codex/langmani-v2-phase2b5-official-demos`.
- Source commit: `8c81058008bb03c4611c9133adf4112e2a21287f`.
- Production branch: `codex/langmani-v2-phase2b6-dataset-production`.
- Local worktree: `D:\LangMani-worktrees\phase2b6-dataset-production`.
- Source authority: Phase 2B.5 `AcceptedOfficialDemoSourcePackage`.
- Frozen specification: `configs/langmani_v2/phase2b6_dataset_production.yaml`.
- Compact evidence: `artifacts/langmani_v2/phase_2b6/`.
- Full arrays, media, archive, and restored bytes: external to Git.

## Frozen identities and totals

| Task | Skill | Task container identity | Episodes | Frames |
| --- | --- | --- | ---: | ---: |
| PickCube-v1 | pick and place | `langmani/official-pickcube-v1` | 1,000 | 77,976 |
| StackCube-v1 | stacking | `langmani/official-stackcube-v1` | 1,000 | 107,420 |
| PushCube-v1 | planar pushing | `langmani/official-pushcube-v1` | 1,000 | 68,978 |
| Total | three skills | `LangManiOfficialMultiSkill-v1` | 3,000 | 254,374 |

Every source identity binds the official revision, ZIP hash, task, source trajectory, reset,
initial state, and action-sequence hash. Every derived identity additionally binds the replay
configuration, camera configuration, deterministic language template, primary split, and dataset
version.

## Ordered execution

1. Verify repository isolation, live upstream SHA, Phase 2B.5 artifacts, source ZIPs, processes,
   GPU state, and peak storage capacity.
2. Copy and lock the frozen specification before the first physical production episode.
3. Validate the JSON/HDF5 schema and native action bytes for all 3,000 source trajectories.
4. Materialize deterministic language assignments and the 700/100/100/50/50 per-task splits.
5. Run the pre-registered post-render visual-shift physical-equivalence pilot.
6. Replay all 3,000 official trajectories and save exactly 254,374 pre-action policy frames.
7. Compute statistics and canonical primary-train-only normalization.
8. Convert each task/split view through the public LeRobot 0.6 writer, then create the 150-episode
   evaluation-only visual-shift roots.
9. Build the immutable unified multi-root package and metadata-only leave-one-skill-out folds.
10. Read every episode through LeRobot 0.6, structurally validate every row, and decode every
    generated video frame.
11. Independently compare every official source action with its derived bytes and run the full
    identity, media, language, visual-lineage, and fold leakage audit.
12. Create a deterministic external archive, restore it without symlinks, rehash every file, and
    run stratified LeRobot loading across every task and primary split.
13. Create the accepted package and Result A only if every gate passes; preserve all training
    authorization flags as false.
14. Validate source/tests/build/evidence, document the exact outcome, commit, push, and verify
    local/upstream/GitHub/server SHA equality.

## Hard stops

Production stops without physical repair if repository isolation, Phase 2B.5 evidence, source
bytes, action arrays, categorical replay agreement, frame/action alignment, privilege exclusion,
primary split disjointness, full readback, archive creation, restore equality, or three-skill
retention fails. Partial bytes and diagnostic identities are preserved.

## Acceptance boundary

Result A may set dataset validation and later ACT/SmolVLA/VLA-JEPA eligibility true. Eligibility is
not authorization. No policy training follows automatically, and throughout Phase 2B.6:

```text
student_policy_training_started = false
optimizer_steps = 0
act_training_started = false
smolvla_training_started = false
vla_jepa_training_started = false
```
