# LangMani 2.0 Phase 2C-A plan

## Objective

Phase 2C-A establishes reproducible traditional imitation-learning baselines on the accepted
`LangManiOfficialMultiSkill-v2` package. It compares three task-specific ACT policies with one
shared three-way Task-ID ACT policy under real closed-loop PickCube, StackCube, and PushCube
simulation.

This phase produces trained checkpoints and physical simulator results. Offline validation loss
is a selection diagnostic, not a policy result.

## Immutable inputs

- Package fingerprint:
  `sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`.
- Accepted policy data: 2,998 episodes and 254,200 frames.
- Training view: 2,099 episodes and 177,718 frames.
- Task training frames: Pick 54,608; Stack 74,891; Push 48,219.
- Frozen exclusions: StackCube source episode 938 and PushCube source episode 202.
- Canonical normalization:
  `LangManiOfficialMultiSkill-v2-primary-train-pooled-natural-frame-v0`.
- Policy observation: base-camera uint8 RGB `256x256` and
  `PandaPolicyStateV0 float32[9]`.
- Action: native Panda `pd_joint_pos float32[8]` at 20 Hz.
- Temporal alignment: `observation[t] -> action[t]`.

The 168-file archive/restore payload remains authoritative. The 169th live-primary file is the
authorized post-finalization package sidecar and is not a changed data payload. No dataset byte,
split, statistic, exclusion, template, archive, or restore artifact may be edited.

## Primary models

| Model | Training view | Conditioning |
| --- | --- | --- |
| Pick ACT | Pick train only | None |
| Stack ACT | Stack train only | None |
| Push ACT | Push train only | None |
| Shared Task-ID ACT | All three train views | Three-way one-hot public ACT ENV token |

All four use the same maintained LeRobot 0.6 ACT core: ResNet-18 without downloaded pretrained
weights, model dimension 512, eight heads, feed-forward dimension 3,200, four encoder layers, one
decoder layer, VAE latent dimension 32, four VAE encoder layers, dropout 0.1, KL weight 10, and
chunk size 16. The shared ENV token is the only structural difference and leaves the Panda state
at nine dimensions.

## Optimization contract

The primary optimizer is AdamW with learning rate `1e-5`, backbone learning rate `1e-5`, weight
decay `1e-4`, no warmup, no scheduler, gradient-norm limit 10, no augmentation, and BF16
autocast. The final common batch size is selected by real RTX 5090 stability probes and then
frozen before primary training.

Each task-specific model targets 20 accepted-frame passes. Strict uniform-task sampling and exact
20 passes are mathematically incompatible for the shared policy because task frame counts differ.
The shared run therefore:

1. preserves strict long-run Pick/Stack/Push sampling of one third each;
2. assigns every task the arithmetic mean of the three task-specific 20-pass sample budgets;
3. records the actual effective passes for every task.

Four immutable checkpoints are retained near 25%, 50%, 75%, and 100% of each planned budget.

## Gates and execution order

1. Independently verify the package erratum, frozen payload, live sidecar, archive, and restore.
2. Audit the clean repository, isolated training environment, installed ACT source, GPU, disk,
   and inactive writer/replay/training processes.
3. Read every train and validation root through the real LeRobot 0.6 API and freeze the complete
   2,099-episode training view.
4. Verify the task-specific views, strict uniform-task sampler, canonical statistics, feature
   allowlist, and H=16 padding masks.
5. Run one full-architecture GPU batch for Pick, Stack, Push, and shared ACT. Require finite loss
   and gradients, masked-padding invariance, one optimizer step, save/reload, and real inference.
6. Run deterministic micro-overfit gates for Pick ACT and shared ACT. These checkpoints are
   pipeline evidence only.
7. Freeze the common stable batch size and all four primary configurations.
8. Train the three task-specific seed-0 policies and shared seed-0 policy.
9. Run padding-masked offline diagnostics at all retained checkpoints. Validation-only selection
   keeps at most one primary candidate per model.
10. Compare execution horizons 1, 4, and 8 on the fixed 30-episode-per-task validation schedules.
    Prefer one common horizon.
11. Apply the basic competence gate. One narrowly scoped contract repair is permitted only if all
    learned policies are zero-success on a task.
12. Freeze checkpoint, processor, normalization, chunk, execution horizon, task condition,
    simulator configuration, and test identities.
13. Run 50 unseen-reset and 50 visual-shift episodes per matching task for each per-task policy
    and for the shared policy.
14. Run the fixed correct/incorrect/shuffled Task-ID intervention for the shared policy.
15. Compute confidence intervals, multi-task interference, latency, smoothness, saturation,
    failure taxonomy, and representative success/failure videos.
16. Independently verify and classify Result A, B, C, or D.

## Fail-closed boundaries

Stop if the package identity changes, another primary-tree difference appears, a frozen exclusion
enters a policy view, leakage appears, padded targets affect loss, a privileged feature reaches a
policy batch, save/reload changes inference, a learned action needs clipping/projection, or a final
setting changes after a test result is visible.

No SmolVLA, VLA-JEPA, language encoder, LoRA, SARM, LatentGuard, expert, PPO teacher, MPC,
reset-replay, or new data-production work is authorized.

## Terminal execution status

The four seed-0 primary runs and all 16 validation-only offline diagnostics completed. The
loss-selected Pick checkpoint then emitted finite but out-of-bounds gripper actions on the first
pre-registered closed-loop infrastructure smoke. The evaluator rejected the chunk before
`env.step`; no clipping or projection was used. Phase 2C-A therefore stopped at step 10 of the
execution plan and is classified `RESULT_D`. Horizon comparison, development evaluation, shared
seed 1, final evaluation, task-condition intervention, and later-policy training were not run.
