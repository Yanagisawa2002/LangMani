# Phase 2B.4-F0 training report

## Outcome

The one authorized micro PPO run completed correctly but did not learn meaningful pushing. This
is algorithm-quality evidence, not an infrastructure failure.

| Item | Observed |
| --- | ---: |
| execution Git | `0740c2fcaf5a34405130f3e3a8a3fac6e9a5d406` |
| GPU | NVIDIA GeForce RTX 5090 |
| vector environments | 128 |
| total environment steps | 1,048,576 |
| optimizer steps | 8,118 |
| elapsed | 944.172 s |
| throughput | 1,110.577 env-step/s |
| resets / completed episodes | 21,162 / 21,034 |
| completed successes | 0 |
| simulator exceptions | 0 |
| peak VRAM | 27.994 MiB allocated by PyTorch |
| peak process memory | 3,501.734 MiB |

The configuration stayed exactly as frozen: 32 rollout steps per environment, 8 minibatches,
4 update epochs, learning rate `3e-4`, gamma `0.99`, GAE lambda `0.95`, clip coefficient `0.2`,
one seed, no sweep, and no resume.

## Learning evidence

Mean first-quartile return was `-5.67375`; the last quartile was `-5.40859`. The increase
`0.26516` was below the frozen `0.5` material-improvement threshold. Mean target-distance progress
changed from `-0.01569 m` to `-0.01346 m`; the `0.00224 m` difference was below the frozen
`0.01 m` threshold and both windows remained net negative. Training success was zero.

The micro checkpoint is 3,755,973 bytes with SHA-256
`3528003bcb9ce1ae12904f0025eb7cffe3edd9836b24e3e4eef6becfcb672bcff`. It remains only under
the target node's ignored `outputs/` tree and is not committed. Reconstruction reproduced the
deterministic action exactly (`max_abs_error=0`) and a real post-reload action was executed.

## Reward-repair decision

Reward revision 0 passed the pre-learning exploitation audit. During evaluation, wrong-object and
workspace failures incurred their declared large negative terminal terms; neither produced a
return comparable to stable success. The run therefore demonstrates failure to learn, not a
profitable reward loophole. The optional one-repair allowance was not used, and no second training
run was started.

No demonstration, archive, failure corpus for training, LeRobot dataset, student statistic, or
student optimizer step was produced.
