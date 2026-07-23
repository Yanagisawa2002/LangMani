# Phase 2B.4-F1 state-centered residual action redesign

The action identity is `state_centered_bounded_residual_v1`. It emits the unchanged native
`float32[8] pd_joint_pos` target.

For normalized current action coordinate `z_current`, policy latent `u`, and explicit residual
scale `s`:

```text
z_next = tanh(atanh_safe(z_current) + s * tanh(u))
action = action_bias + action_scale * z_next
```

The first seven current coordinates come from Panda arm qpos. Mean finger qpos is mapped from
`[0, 0.04]` to the native gripper coordinate. Arm residual scales are 0.08; the gripper scale is
0.25. Zero residual reproduces the current target except for the declared numerical guard within
`1e-6` of a coordinate boundary; the declared zero-residual tolerance is `2e-6`.

The derivative used in PPO likelihood is the product of:

```text
action_scale
* (1 - z_next^2)
* residual_scale
* (1 - tanh(u)^2)
```

Rollout, update, deterministic evaluation, and checkpoint reconstruction call the same transform.
There is no emitted-action clipping, projection, rejection sampling, invalid-action replacement,
expert fallback, or planner. Before any probe, 100,000 sampled actions must be finite and in bounds
and a paired real-physics comparison must demonstrate materially safer local exploration.

The final 100,000-action audit passed with zero non-finite or out-of-bound actions, zero clipping or
projection, and maximum zero-residual error `1.0132789611816406e-6` against tolerance `2e-6`.

On 32 identical reset/noise pairs over eight real physics steps, mean maximum arm target
displacement fell from 0.712 to 0.305 rad and mean maximum TCP step translation fell from 0.0902 to
0.0448 m. Locally smooth prefixes increased from 0/32 to 21/32. Both mappings had zero
wrong-object or workspace events in this short prefix, so the evidence establishes safer local
kinematics, not a lower observed short-prefix safety-event rate or task competence.
