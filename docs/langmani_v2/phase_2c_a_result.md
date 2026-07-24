# LangMani 2.0 Phase 2C-A result

Phase 2C-A is **Result D — training/consumer pipeline invalid**. Four real ACT runs completed and
all 16 checkpoints passed offline diagnostics, but the loss-selected Pick ACT emitted native
`pd_joint_pos` gripper commands above 1.0 on the first closed-loop infrastructure smoke. The
evaluator rejected the chunk before any environment step. This is not a policy-quality result.

## Recruiter-readable summary

1. The accepted 2,998-episode/254,200-frame multi-skill package was used without changing any
   dataset, split, statistic, exclusion, archive, restore, or sidecar byte.
2. Pick, Stack, Push, and shared three-way Task-ID ACT seed-0 models were trained with the same
   LeRobot 0.6 ACT core.
3. Padding was handled by explicit `action_is_pad`; padded targets had zero action-loss weight.
4. Per-task closed-loop performance is unavailable because the infrastructure smoke stopped before
   the first action.
5. Shared closed-loop performance is unavailable; shared seed 1 was not started.
6. Multi-task interference was not measured.
7. Task-conditioning sensitivity was not measured.
8. Visual-shift performance was not measured.
9. The observed failure mode was finite but out-of-bounds gripper output from the selected Pick
   checkpoint.
10. SmolVLA is not justified or authorized by this Result D package.

The immutable state is:

```text
training_complete=true
offline_diagnostics_complete=true
closed_loop_development_started=false
final_evaluation_started=false
act_baselines_validated=false
smolvla_phase_eligible=false
smolvla_training_authorized=false
vla_jepa_training_authorized=false
```

Recommended next work is a separately authorized pre-policy action-validity investigation. It
should determine why mean/std-regressed gripper outputs cross native bounds and pre-register a
lawful action representation or eligibility gate before any new training. It must not retroactively
clip these checkpoints, reinterpret Result D as quality evidence, or start SmolVLA/VLA-JEPA.
