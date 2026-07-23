# Phase 2B.4-F0 PPO implementation audit

## Audited upstream

The compatible installed stack is ManiSkill 3.0.1, SAPIEN 3.0.3, Python 3.12, PyTorch 2.11.0,
CUDA 12.8, Gymnasium 1.2.3, and NumPy 2.2.6. The maintained reference is ManiSkill tag `v3.0.1`,
commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`, specifically
`examples/baselines/ppo/ppo.py` and its `baselines.sh` launcher.

The reference assumes state observations and GPU PhysX vectorization. It constructs a
`ManiSkillVectorEnv`, uses separate actor and critic MLPs with three 256-unit tanh hidden layers,
stores batched rollout tensors, computes GAE, applies the clipped PPO surrogate, and optimizes
with Adam. Evaluation uses the actor mean. Its checkpoint contains the agent state dict and its
example does not implement a complete optimizer/RNG resume.

The maintained script samples an unbounded Normal action and clamps it before `env.step`. That
semantic is incompatible with F0's native-action requirement and is not copied.

## Narrow adaptation

F0 retains the upstream network shape, rollout layout, GAE structure, PPO loss, Adam update, and
deterministic mean-policy idea. The project-owned changes are explicit:

1. one Gaussian-latent/tanh/affine action distribution replaces Normal-plus-clamp;
2. exact bounds come from the active environment;
3. the 87D current privileged state replaces generic flattened observations;
4. fixed per-component scales replace fitted or online observation statistics;
5. manual asynchronous reset preserves explicit project seed namespaces;
6. truncation bootstraps value while termination does not;
7. the checkpoint includes policy, optimizer, configuration, contract fingerprints, global step,
   and CPU/CUDA RNG states;
8. F0 exposes checkpoint reconstruction, not training resume;
9. deterministic evaluation uses one real physical environment and the identical action transform.

ManiSkill 3.0.1 replaces its complete episode-seed array when an explicit seed list accompanies a
partial reset. The adaptation therefore maintains the complete current seed vector and submits it
with every masked reset. The simulator reset mask still restricts physical changes to selected
slots. The vector audit independently checks that untouched object state is byte-exact, selected
state changes, seeds are unique, and elapsed counters reset.

No source is copied from ManiSkill. The upstream revision and every semantic difference are
recorded in the machine-readable implementation audit.
