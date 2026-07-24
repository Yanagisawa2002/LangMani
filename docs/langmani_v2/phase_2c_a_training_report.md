# Phase 2C-A ACT training report

## Data and configuration

The canonical package fingerprint is
`sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`.
The immutable training views contain 54,608 Pick, 74,891 Stack, and 48,219 Push frames. All roots
loaded through LeRobot 0.6, both frozen exclusions remained absent, images decoded, state/action
shapes were `[9]`/`[8]`, and no privileged field entered a policy batch.

All models used batch 408, BF16, AdamW, learning rate and backbone learning rate `1e-5`, weight
decay `1e-4`, gradient limit 10, no warmup, no scheduler, no augmentation, and four checkpoints.
Pick/Stack/Push consumed exactly 20 frame passes. Shared used strict 1:1:1 sampling, 136 examples
per task per batch, and exactly 1,184,832 examples per task.

GPU smoke passed for all four models. Per-task peak allocated/reserved memory was
25.025/28.236 GiB; shared was 25.147/28.324 GiB. Pick and shared 200-step micro-overfit both
decreased total loss and unpadded raw action error, with bit-exact checkpoint reload output.

## Primary runs

| Model | Steps | Effective samples | Wall time | Mean examples/s | Final checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| Pick ACT | 2,680 | 1,092,160 Pick | 2,465.05 s | 2,164.95 | `sha256:da7128…e0f3` |
| Stack ACT | 3,680 | 1,497,820 Stack | 3,392.13 s | 2,157.81 | `sha256:9458aa…5a61` |
| Push ACT | 2,380 | 964,380 Push | 2,180.34 s | 2,156.44 | `sha256:6b51e3…67bd` |
| Shared seed 0 | 8,712 | 1,184,832 per task | 7,782.34 s | 2,183.69 | `sha256:ada3cb…d870` |

All 16 checkpoint directories contain complete model/processor/optimizer/RNG manifests. The
independent verifier rehashes every listed checkpoint file. An earlier Pick run at commit
`dc402fb` processed 1,091,000 rather than the exact 1,092,160 target because its step formula
ignored the retained partial batch. It remains a rejected diagnostic and is not included above.

The source-controlled compact registry is
`artifacts/langmani_v2/phase_2c_a/training_summary.json`; full generated metrics remain below
`/root/autodl-tmp/langmani-phase2c-a/training-v2`.
