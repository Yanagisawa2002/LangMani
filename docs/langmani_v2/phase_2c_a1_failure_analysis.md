# LangMani 2.0 Phase 2C-A.1 failure analysis

## Classification

Phase 2C-A.1 is `RESULT_B`: infrastructure, serialization, action bounds,
checkpoint screening, and the complete simulator execution chain passed, but
policy quality was weak. This differs from Phase 2C-A `RESULT_D`, where the
selected unbounded policy's first chunk was rejected before `env.step`.

## What is now proven

- The canonical 64-character dataset identity and frozen payload tree remained
  unchanged.
- The bounded head, physical-action loss, save/reload path, and runtime
  processors agree.
- All 16 checkpoints survived 10,000-query validation action audits.
- The selected Pick policy reached real `env.step` during the smoke.
- Development, Task-ID intervention, unseen-reset, and visual-shift schedules
  all completed.
- Across the 540 development and final episodes (180 plus 360), there were
  zero invalid actions and zero simulator errors.
- The final 360 episodes executed 18,000 native actions without clipping or
  projection.

The signal interruption observed during repository transfer did not interrupt
training or evaluation and is not an experimental failure category.

## Observed failure taxonomy

The 360 final episodes all timed out. Pooled categories were:

| Policy | Task | Failure categories |
| --- | --- | --- |
| Per-task | Pick | 53 no initial motion, 6 failed grasp, 1 object drop |
| Per-task | Stack | 58 no initial motion, 2 failed grasp |
| Per-task | Push | 60 no initial motion |
| Shared | Pick | 60 no initial motion |
| Shared | Stack | 60 no initial motion |
| Shared | Push | 58 no initial motion, 2 insufficient push |

The dominant observed mode is insufficient task-directed motion, followed by a
small number of grasp/drop/push failures. No wrong-object interaction,
out-of-bound action, non-finite action, or simulator exception was observed.

## Supported diagnosis

The bounded head solved the exact Phase 2C-A action-validity defect, but the
frozen ACT training recipe did not learn useful closed-loop behavior for these
official multi-skill demonstrations. Offline loss reduction, checkpoint
screening, bounded outputs, and task-ID sensitivity were insufficient
predictors of control success.

The evidence does not isolate a single architectural cause. Plausible causes
such as behavior-cloning covariate shift, horizon/chunk mismatch, weak visual
state grounding, or physical-coordinate regression difficulty remain
hypotheses. They were not separately ablated and must not be presented as
proven.

## Visual shift and interference limits

Both unseen-reset and visual-shift results are 0/30 in every group. The measured
visual degradation of zero is a floor equality, not robustness evidence.
Likewise, shared-minus-per-task success differences of zero do not show that
multi-skill interference is absent because every comparator is already at
zero.

## Stop decision

The authorized milestone explicitly permits only one final ACT repair. The
consumer pipeline is valid and the remaining failure is policy quality, so
another ACT architecture, seed, hyperparameter search, checkpoint reselection,
or dataset change is not authorized. `act_phase_closed=true` and
`further_act_architecture_authorized=false`.
