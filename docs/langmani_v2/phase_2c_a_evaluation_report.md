# Phase 2C-A evaluation report

## Offline validation

All 16 retained checkpoints reloaded and completed validation-only, padding-masked diagnostics.
Every output was finite and nonconstant. The frozen loss rule selected the final checkpoint for
each model:

| Model | Selected step | Masked ACT loss | Unpadded raw L1 | First-action raw L1 |
| --- | ---: | ---: | ---: | ---: |
| Pick | 2,680 | 0.161432 | 0.038714 | 0.021126 |
| Stack | 3,680 | 0.123476 | 0.044875 | 0.024702 |
| Push | 2,380 | 0.082047 | 0.012363 | 0.009001 |
| Shared seed 0 | 8,712 | 0.107751 | 0.032001 | 0.018198 |

Shared selected raw L1 was 0.032349 Pick, 0.044145 Stack, and 0.012217 Push. These are offline
diagnostics, not success results.

## Closed-loop hard stop

Before the pre-registered H=1/4/8 development matrix, one Pick/H=4 validation episode ran as an
infrastructure smoke. The selected checkpoint reloaded and produced a finite `16x8` chunk. All
seven arm dimensions were within the exact live `pd_joint_pos` action space, but every gripper
value was above the native high bound 1.0; the first four were 1.051940, 1.055561, 1.059146, and
1.054304. The evaluator recorded one policy query, zero executed actions, zero environment steps,
zero simulator errors, and `invalid_policy_output`.

Declared Panda bounds were compared element by element with the live Gymnasium action space and
matched exactly. A diagnostic replay found that the unselected step-670 Pick checkpoint produced a
valid first H=4 chunk on the same reset. It was not substituted after observing the smoke because
that would replace the already locked validation-loss selection with outcome-informed selection.
No clipping, projection, binary conversion, or checkpoint reselection was performed.

The hard stop makes the following metrics unavailable, not zero: horizon comparison, 30-episode
validation success, unseen-reset success, visual-shift success, confidence intervals, latency
distribution, representative videos, and final-test results. No final identity was opened.
