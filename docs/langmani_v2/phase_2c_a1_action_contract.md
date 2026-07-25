# LangMani 2.0 Phase 2C-A.1 action contract

## Native interface

All three official environments expose the same action interface:

```text
control mode: pd_joint_pos
frequency:    20 Hz
dtype:        float32
shape:        [8]
```

The task-consistent native bounds observed from the actual ManiSkill action
spaces are:

```text
lower = [
  -2.8973000049591064,
  -1.7627999782562256,
  -2.8973000049591064,
  -3.0717999935150146,
  -2.8973000049591064,
  -0.017500000074505806,
  -2.8973000049591064,
  -1.0
]

upper = [
  2.8973000049591064,
  1.7627999782562256,
  2.8973000049591064,
  -0.0697999969124794,
  2.8973000049591064,
  3.752500057220459,
  2.8973000049591064,
  1.0
]
```

The machine-readable bounds manifest fingerprint is
`sha256:9838da7b5e929b8390e3b3353366ad33988529e785fae48baf6068207a7d2b06`.

## Policy mapping

`bounded_action_head_v1` is the only authorized output formulation:

```text
u = Linear(decoder_token)
z = tanh(float32(u))
alpha = 0.5 * (z + 1)
a = lerp(lower_float32, upper_float32, alpha)
```

`torch.lerp` is the numerically stable float32 implementation of
`lower + 0.5 * (z + 1) * (upper - lower)`. For every finite `u`, `alpha` is in
`[0,1]` and each physical action component lies inside its native interval.
The lower and upper vectors are persistent model buffers and are verified on
reload.

The bounded-head semantic fingerprint is
`sha256:6462a774ad6dad33d5f3853b056d16c0c63f18e882b026ea7329b708119a2fa3`.

## Training and inference equivalence

Actions remain in physical `pd_joint_pos` coordinates throughout the repaired
model path:

```text
dataset physical action
-> action preprocessor IDENTITY
-> bounded ACT physical prediction
-> padding-masked physical L1 + unchanged KL
-> action postprocessor IDENTITY
-> ActionChunk
-> evaluator native-bound gate
-> environment
```

State and image normalization are unchanged. Only the action normalization mode
changes from `MEAN_STD` to `IDENTITY` because the trained head itself now emits
physical actions. Training and deployment therefore use one representation.

For target `y`, prediction `a`, and padding mask `p`, the action term is:

```text
sum(abs(a - y) * not(p)) / (8 * count(not(p)))
```

An all-padding synthetic batch returns a finite zero action loss. Padded
timesteps have no gradient contribution to the action term. The existing ACT KL
term and all other architecture and optimizer settings are unchanged.

## Forbidden corrections

The contract contains no:

- post-hoc clamp or clip;
- projection;
- rejection sampling;
- zero, previous, or expert action replacement;
- gripper threshold or binary conversion;
- processor-specific gripper scaling;
- evaluator repair;
- reliance on environment correction.

The evaluator remains an independent hard gate. It rejects malformed,
non-finite, or out-of-bound policy actions; it never alters them.

## Current pre-evaluation evidence

The independent random audit covered 100,000 raw action chunks, 16 actions per
chunk, and 1,600,000 total actions. It observed zero non-finite values, zero
lower or upper violations, zero clipping events, and zero projection events.
Its fingerprint is
`sha256:16e67f6c8c7b0889d4888c00e8d05cc448407b785014bd98c7939da5b70ba048`.

The GPU smoke and both real-data micro-overfits passed bounds, padding, and
save/reload checks. These are pipeline checks only. Policy quality remains
unmeasured until the authorized closed-loop evaluations complete.
