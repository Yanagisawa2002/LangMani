# Phase 2B.4-F0 privileged-state PPO teacher feasibility plan

## Scope and immutable ancestry

This bounded study starts only from
`3ded856bc0fba7cd749f1fb0caf738fd7fce5751` on
`codex/langmani-v2-phase2b3-reset-replay-equivalence`. The independent geometry commit
`ae6aae49c61d96c68de434e08dc1167461a29543` is excluded. The simulator-MPC Result C and
reset-replay Result D packages are immutable inputs, not implementation sources.

F0 asks whether one privileged current-state PPO teacher can learn safe closed-loop pushing and
whether its pipeline is trustworthy enough to make later full training merely eligible. It does
not authorize that later training.

## Frozen implementation

- Environment: unchanged `LangMani-PushToRegion-v0`.
- Control: native `pd_joint_pos`, `float32[8]`, exact environment bounds.
- Actor/critic: separate three-hidden-layer 256-unit tanh MLPs.
- Distribution: Gaussian latent, tanh, then affine map to exact bounds. The same transform owns
  stochastic rollout, PPO likelihoods, deterministic evaluation, and checkpoint reconstruction.
- Observation: 87-dimensional current privileged state described by the committed manifest.
- Normalization: fixed componentwise scales, no fitted statistics and no clipping.
- Reward: revision 0 described by the committed reward manifest. At most one specifically
  justified repair is available before micro training; it is not pre-authorized.
- Vectorization: 128 GPU environments, explicit disjoint seeds, masked partial reset, no motion
  planner.
- PPO budget: exactly 1,048,576 environment steps, 32 steps per environment per rollout, one
  configuration, no sweep and no resume command.

## Frozen evaluation

Micro evaluation uses 48 seeds `800000..800047`, equally cycling cube/cylinder and
left/forward-right Standard tasks. It requires at least 70% overall, 70% cube, 50% cylinder, one
success in both directions, material training-return and target-progress improvement, nonzero
training success, deterministic reload, nonconstant legal actions, and zero safety events.

Only a passing micro gate opens the 60-episode feasibility-development evaluation on seeds
`810000..810059`. It covers both objects, all four directions, Standard and a bounded real Hard
subset. It requires at least 50% overall, 60% cube, 30% cylinder, one success per canonical
direction, and zero safety events.

Formal seeds `66300..66399` and future full-development seeds `820000..919999` are sealed.

## Stop conditions

F0 stops as Result D for invalid vector semantics, exploitable reward, illegal actions,
non-finite optimization, or inconsistent checkpoint reconstruction. It stops as Result C when the
frozen micro gate fails after the optional single reward repair. It returns Result B when micro
passes but broader development fails, and Result A only when every pipeline, micro, development,
and safety gate passes.

Regardless of result, full PPO training, expert qualification, collection, SmolVLA, and student
training remain unauthorized. F0 produces no demonstration, dataset, or student normalization
statistics.
