# Phase 2B.4-F0 result

## RESULT_C — PPO fails to learn pushing in the bounded micro study

The technical pipeline is valid: repository isolation, prior evidence hashes, native target
installation, 128-environment vector semantics, reward audit, 100,000-action audit, PPO update,
finite gradients, checkpoint reconstruction, deterministic inference, and real environment
execution all passed. The learned behavior did not.

The frozen micro run had zero training successes and no material return or target-progress
improvement. Its deterministic evaluation reached 0/48 and produced 22 wrong-object displacements
plus 12 workspace exits. The hard stop applies before broader development.

This is not evidence that reinforcement learning cannot solve the task. It is evidence that this
specific observation/action/reward/PPO configuration and one-million-step budget failed the
declared feasibility gate.

## Answers to the phase questions

1. The geometry route stopped after 2/12 diagnostic successes and one zero-tolerance wrong-object
   displacement. Simulator MPC stopped because public snapshot restore differed materially from
   live contact continuation. Reset-replay stopped because original executable action prefixes
   were unavailable; it made no physical nondeterminism claim.
2. The implementation adapts ManiSkill 3.0.1 PPO at
   `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
3. The teacher receives an 87D current privileged state: Panda qpos/qvel and TCP pose; both object
   poses and velocities; target/goal geometry, identities, displacements and containment;
   distractor-relative vectors; contact, lift/alignment, workspace margins, and elapsed time.
4. The future student remains exactly base-camera RGB plus `PandaPolicyStateV0[9]`; the schema
   checker rejects every other teacher field.
5. A Gaussian latent passes through tanh and the exact environment-bound affine map. Rollout,
   likelihood calculation, reload, and deterministic evaluation share it.
6. Reward revision 0 uses step cost, useful-side approach, target progress, containment progress,
   contact-side terms, action regularization, stable-success bonus, and explicit safety penalties.
7. The constructed reward-hacking audit passed; no repair was used.
8. The 128-environment audit passed with unique seeds, independent partial reset, exact untouched
   state, correct counters, finite state, and no planner calls.
9. PPO did not learn the micro distribution.
10. Neither cube nor cylinder succeeded.
11. Broader generalization is unverified because the micro hard stop prevented that evaluation.
12. Evaluation observed 22 wrong-object displacements and 12 target workspace exits.
13. Full PPO training is not eligible.
14. Qualification, collection, and SmolVLA remain unauthorized because F0 failed its micro gate
    and no accepted demonstration source exists.
15. The recommended next milestone is a separately authorized design-only
    **Phase 2B.4-F1 — PPO Teacher Failure Diagnosis and Safe Curriculum Redesign**. It should
    inspect action initialization/exploration, early termination composition, contact-credit
    assignment, and a safe staged curriculum without running another training job until a new
    bounded protocol is explicitly approved.

## Final authorization

```text
ppo_teacher_feasibility_validated = false
ppo_full_training_eligible = false
ppo_full_training_authorized = false
expert_qualification_authorized = false
data_collection_authorized = false
smolvla_training_authorized = false
training_started_for_student_policy = false
demonstration_source_validated = false
```
