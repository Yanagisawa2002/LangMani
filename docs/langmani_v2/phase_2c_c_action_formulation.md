# Phase 2C-C Pick action formulation

Phase 2C-C asks one question: did the frozen ACT and Pick SmolVLA policies fail because absolute
joint-position targets are too brittle for closed-loop behavior cloning?

The comparison starts from the immutable Phase 2C-B Pick checkpoint and trains exactly one
relative-action Pick SmolVLA. Data, observations, instructions, task definitions, success,
official model revision, optimization, seed, precision, and training length remain unchanged.

## Public state and native action

`PandaPolicyStateV0[9]` is ordered as:

1. `panda_joint1` through `panda_joint7`, measured in radians;
2. `panda_finger_joint1`;
3. `panda_finger_joint2`.

There is no additional state field. Native `action[0:7]` contains absolute arm joint-position
targets in radians. Native `action[7]` is the Panda mimic controller's normalized gripper command
in `[-1,1]`; it is not a raw metre-valued joint target.

The current gripper reference is the arithmetic mean of the two measured finger positions, mapped
from the controller's physical `[-0.01,0.04]` metre range to `[-1,1]`. This makes the current native
action reference deterministic from public state alone.

## Derived view

The metadata/action-only view records:

```text
delta_action[t] =
physical_target_action[t] - current_action_reference(observation.state[t])
```

It stores no image bytes and changes no state, instruction, timestamp, episode, split, frame, or
media identity. The current observation at a training/query anchor is used for the complete
50-action chunk. Future state is not read by the model or transform.

## Bounded model target and deployment

The model target is a safe-logit residual, not an unconstrained physical delta:

```text
z_current = physical_to_normalized(current_action_reference)
d = atanh_safe(z_target) - atanh_safe(z_current)
u_target = atanh(d / residual_scale)

z_next = tanh(atanh_safe(z_current) + residual_scale * tanh(u_model))
physical_action = normalized_to_physical(z_next)
```

`atanh_safe(z)` uses `z * (1 - 2^-24)`. One train-derived arm scale is shared by all seven arm
components and one scale is used for the normalized gripper component:

```text
arm scale:     0.07757549732923508
gripper scale: 16.947166442871094
```

Both are exactly 1.01 times the corresponding train-only maximum safe-logit displacement after
float32 rounding. They are frozen before training. No clipping, projection, action replacement,
or scale sweep exists.

## Gates

Training is blocked until all of these pass:

- the source/dataset/Phase 2C-A.1/Phase 2C-B/checkpoint identities;
- exact recomputation of the two frozen scales;
- 100,000 random 50-action chunks with no non-finite value or bound violation;
- zero-residual identity within `1e-6`;
- reconstruction of every accepted Pick train, validation, and unseen-reset frame within `1e-6`;
- exact transform and checkpoint save/reload.

The final unseen-reset schedule stays closed unless the relative policy reaches at least 3/30
validation successes.
