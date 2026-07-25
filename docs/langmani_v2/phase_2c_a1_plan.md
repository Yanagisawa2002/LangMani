# LangMani 2.0 Phase 2C-A.1 plan

## Objective

Phase 2C-A.1 is the single final ACT repair. It replaces the invalid unbounded
consumer path with one structurally bounded action head, retrains the same four
seed-0 ACT baselines once, obtains real closed-loop results, measures
multi-skill interference, and permanently closes ACT.

This phase does not authorize SmolVLA or VLA-JEPA.

## Frozen inputs

- source commit:
  `135f13d11f0f47173abe42fa19d969e519d7b1d9`;
- dataset: `LangManiOfficialMultiSkill-v2`;
- package fingerprint:
  `sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`;
- accepted episodes/frames: 2,998 / 254,200;
- primary training episodes: Pick 700, Stack 700, Push 699;
- action chunk: 16 at 20 Hz with explicit `action_is_pad`;
- model seed: 0 only;
- four models: Pick, Stack, Push, and shared Task-ID ACT;
- shared sampler: uniform 1/3 per task;
- effective-sample budgets inherited unchanged from Phase 2C-A.

Dataset bytes, exclusions, splits, media, normalization source views, language
templates, visual shifts, action padding, task definitions, and environment
contracts remain frozen.

## Ordered gates

1. Rehash Phase 2C-A evidence, all 16 prior checkpoints, the accepted package,
   and the 168/169-file identity resolution.
2. Trace the complete old action path and classify the pre-step overflow.
3. Verify identical native action bounds for all three official tasks.
4. Implement only `bounded_action_head_v1` and the minimum serialization
   identity required to reload it.
5. Prove bounds over 100,000 random raw chunks and verify padding-loss
   behavior.
6. Run one real LeRobot/GPU forward, backward, optimizer, save, reload, and
   inference smoke.
7. Run deterministic real-data micro-overfits for Pick and shared ACT.
8. Train exactly four full seed-0 models using the frozen budgets.
9. Rehash and screen all 16 retained checkpoints. Each checkpoint must complete
   a 10,000-query action-contract audit over ordered validation observations.
10. Select at most two eligible checkpoints per model using contract filters,
    degeneration filters, and a four-metric validation rank.
11. Require one Pick action to pass the native bound gate and reach a real
    `env.step`.
12. Compare `H_exec` 1, 4, and 8 on one fixed validation subset; freeze one
    common horizon when practical.
13. Run 30 matching validation episodes for each per-task model and 30 per task
    for the shared model.
14. Run correct, wrong, and shuffled Task-ID interventions on a fixed shared
    validation subset.
15. Freeze checkpoint, processors, horizon, evaluator, and final identities,
    then run unseen-reset and visual-shift test schedules exactly once.
16. Report confidence intervals, failure taxonomy, latency, visual degradation,
    and shared-minus-per-task interference.
17. Run an independent verifier, isolated package builds, static checks, and
    repository/remote identity checks.
18. Classify Result A/B/C/D and set `act_phase_closed=true` in every case.

Before any closed-loop result is observed, this run further freezes the
following bounded choices:

- checkpoint retention is one validation-ranked checkpoint per model
  (`maximum=1`);
- the horizon comparison uses the first five rows of the immutable
  PickCube-v1 validation schedule:
  `phase2c-a:validation:pickcube:000` through
  `phase2c-a:validation:pickcube:004`;
- the Task-ID intervention uses the first five immutable validation rows per
  task, with seed `20260725`, the cyclic-next wrong-ID rule, and the
  nondegenerate balanced shuffled-ID rule;
- the official evaluation-schedule fingerprint is
  `sha256:096e657348beb58cb81ffeecfd2627b5ef36dfc3969cece246a7df5f1b15223f`.

These choices use validation identities only and cannot change after their
results are visible.

## Hard stops

Stop the active stage if any dataset identity changes, excluded input enters a
view, padding contributes to loss, privileged state reaches the policy, the
bounded mapping or reload semantics drift, clipping/projection appears, no
action reaches `env.step`, or a final configuration changes after test results
are visible.

An infrastructure interruption may resume the same immutable run or repeat only
the missing artifact. It does not authorize another model seed, sweep, data
stage, ACT architecture, SmolVLA, or VLA-JEPA.

## Result interpretation

Low or zero policy success is a valid model-quality result after the consumer
pipeline is physically valid. Results A, B, and C make a later separately
authorized SmolVLA phase eligible but do not authorize training. Result D closes
ACT and leaves SmolVLA ineligible until the generic policy/action contract is
resolved outside another ACT repair.
