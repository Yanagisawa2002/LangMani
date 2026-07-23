# Phase 2B.4-F0 reward audit

## Revision 0

The frozen reward uses only consecutive current physical states and unchanged environment
metrics:

- `-0.01` per step;
- `20 * (previous target distance - current target distance)`;
- `2 *` useful behind-point approach progress;
- `2 *` containment-fraction progress;
- a maximum `0.05` bonus for first target contact from the useful side;
- a maximum `-0.10` penalty for target contact from the wrong side;
- small squared action-change and near-saturation penalties;
- `+10` only for the unchanged stable-success predicate;
- explicit negative terms for wrong-object contact/displacement, workspace exit, lift, topple,
  grasp, invalid/out-of-bounds action, and robot collision.

No term teleports, constrains, or directly commands an object. Potential differences do not
accumulate reward for hovering. The reward is frozen before learning with revision count zero.

## Constructed behavior audit

The deterministic diagnostic totals for revision 0 are:

| Behavior | Return |
| --- | ---: |
| no movement for 20 steps | -0.20 |
| hover after one approach improvement | -0.04 |
| wrong-side approach/contact | -0.21 |
| repeated tapping | -0.15 |
| move away | -1.61 |
| wrong-object contact plus displacement | -20.01 |
| progress followed by workspace exit | -10.01 |
| lift | -12.01 |
| topple | -12.01 |
| oscillation | -0.712 |
| saturated actions for 20 steps | -0.60 |
| brief unstable containment without success | 7.79 |
| canonical stable success | 19.28 |

Canonical success is highest; the strongest unsuccessful construction is below 75% of it;
wrong-object and workspace violations are negative; hovering cannot dominate progress; and
oscillation is negative. The pre-learning audit passes, so no reward repair is used or justified.
