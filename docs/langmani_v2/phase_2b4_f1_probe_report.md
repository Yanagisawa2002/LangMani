# Phase 2B.4-F1 probe report

## Probe A configuration and training

Probe A used the frozen Stage-0 cube-only Standard left/forward-right distribution, current-state
residual action transform, reward revision 0, seed 240410, 128 environments, 32 rollout steps, four
PPO epochs, eight minibatches, and exactly 262,144 environment steps. It ran for 121.951 seconds at
2,149.59 environment steps/s, performed 2,048 optimizer steps, and saved/reloaded checkpoint
`sha256:7e420093e7a6e09baf10c9d8e39525f207c8de416b9adecab4a94a600a62ab95`
with zero deterministic-action error.

Training completed 6,327 episodes and observed zero success; all completed episodes reached a
training safety termination. Mean episode return deteriorated from -4.001 in the first quartile to
-7.472 in the last. No resume, sweep, or second seed was used.

## Fixed 32-episode evaluation

- stable success: 0/32;
- useful approach entry: 25/32;
- correct target contact: 0/32;
- target-directed progress: 0/32;
- wrong-object interaction: 7;
- target workspace exit: 0;
- outcomes: 25 timeouts and seven wrong-object terminations;
- invalid, non-finite, or out-of-bound action: 0;
- clipping or projection: 0;
- mean episode length: 202.31 steps.

Probe A failed the zero-tolerance wrong-object gate, 25% contact gate, and 20% progress gate. Probe
B was not authorized, configured for execution, or run. Combined F1 training expenditure was
262,144 of the 524,288-step ceiling.
