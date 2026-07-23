# Phase 2B.4-F0 PPO audit

The authoritative implementation audit is
[`phase_2b4_f0_ppo_implementation_audit.md`](phase_2b4_f0_ppo_implementation_audit.md).

The key decision is to retain the maintained ManiSkill 3.0.1 PPO/GAE/vectorized-training
structure while rejecting its unbounded-Normal-plus-clamp action path. F0 instead uses a single
structurally bounded tanh-affine distribution throughout rollout, update, checkpoint reload, and
deterministic physical evaluation. No action projection, replacement, rejection sampling, or
expert fallback exists.
