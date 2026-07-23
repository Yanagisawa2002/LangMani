# LangMani 2.0 Phase 2B.4-F1 plan

## Research question

F1 asks why the technically valid F0 PPO pipeline produced zero successful pushes and whether one
state-centered, structurally bounded exploration redesign can recover safe Stage-0 competence. It
is the final bounded PPO-specific diagnostic, not full training or teacher qualification.

## Immutable inputs and isolation

- Source is F0 Result C commit `c5892a357da85f753d5d0c3775ccf2b94e0d8910`.
- Geometry commit `ae6aae49c61d96c68de434e08dc1167461a29543` is excluded.
- The F0 compact package and checkpoint hash are read-only inputs.
- Formal seeds 66300--66399 and F0 evaluation identities are never reused.
- Collection, LeRobot, SmolVLA, ACT/VLA-JEPA, student optimization, Stage 1/2, and full PPO are
  prohibited.

## Ordered execution

1. Verify repository isolation and every F0 hash.
2. Reconstruct only retained F0 learning metrics and enumerate unavailable fields.
3. Verify termination, truncation, bootstrap, and asynchronous-reset semantics.
4. Audit at least 100,000 initial actions and the frozen F0 checkpoint on new diagnostic seeds.
5. Validate the residual transform and its PPO likelihood.
6. Compare F0 and residual initial policies on identical resets and random draws for 32 eight-step
   physical prefixes.
7. Stop as Result C if residual exploration is not materially safer.
8. Freeze reward decision and Probe A; run exactly 262,144 steps.
9. Evaluate 32 disjoint episodes and stop on any failed Probe A gate.
10. Run one 262,144-step Probe B only if Probe A authorizes it; evaluate 48 disjoint episodes.
11. Classify A/B/C/D, independently verify compact evidence, test/build, commit, and push.

The combined new training budget is at most 524,288 environment steps. There is no Probe C.

## Mandatory final state

`ppo_full_training_authorized`, `expert_qualification_authorized`,
`data_collection_authorized`, `smolvla_training_authorized`,
`demonstration_source_validated`, and `student_policy_training_started` remain false for every
result. Result A may set only `ppo_safe_curriculum_eligible=true`.
