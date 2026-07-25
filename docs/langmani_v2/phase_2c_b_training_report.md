# LangMani 2.0 Phase 2C-B training report

## Scope and inputs

Phase 2C-B evaluated the official pretrained SmolVLA family after the bounded ACT phase closed as
Result B. It consumed only the immutable `LangManiOfficialMultiSkill-v2` package identified by
`sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`.
The package contains 2,998 episodes and 254,200 frames across Pick, Stack, and Push. No dataset,
split, normalization, episode, task, or success definition changed.

The official base was `lerobot/smolvla_base` at revision
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`, with
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at revision
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Strict loading reconstructed all
450,046,176 parameters, with 99,880,992 trainable and 350,165,184 frozen parameters and no
reinitialized tensor.

## Embodiment and action path

The public embodiment uses one RGB camera, `PandaPolicyStateV0[9]`, and native
`pd_joint_pos[8]`. SmolVLA's published 32-dimensional internal state/action padding remains
unchanged. Dataset camera bytes are converted once from `uint8` to `float32` in `[0,1]` before
official preprocessing.

`smolvla_bounded_action_latent_v1` is the only physical action representation:

```text
physical target -> affine [-1,1] -> (1-epsilon) -> atanh -> official flow target
official latent -> tanh -> affine native Panda bounds -> physical action
```

`epsilon=1e-6`. The 100,000-chunk/1,600,000-action audit produced zero non-finite or invalid
actions and zero clipping or projection events. Processor and checkpoint serialization were
strictly reconstructed.

## Staged checks

The real GPU smoke completed one optimizer step, one model query, and one real simulator
`env.step`, with no invalid action or simulator error. The Pick and shared 500-step micro-overfits
both passed:

| Stage | Initial loss | Final loss | Initial action MAE | Final action MAE |
| --- | ---: | ---: | ---: | ---: |
| Pick micro | 9.4365 | 0.3652 | 1.0173 | 0.1369 |
| Shared micro | 15.0239 | 0.2604 | 1.2887 | 0.1788 |

The shared micro model changed its output across different instructions
(`max_abs_difference=1.921348`). This proves condition sensitivity in the bounded micro fixture,
not correct language-conditioned control or closed-loop competence.

## Full Pick training

Only Model P was authorized for full training. It used 700 Pick train episodes and 54,608 frames,
seed 0, a 50-step action chunk, batch 4 with four-step gradient accumulation, BF16, AdamW at
`1e-4`, 1,000 warmup steps, cosine decay, and 20,000 optimizer steps.

The authoritative run was produced at Git commit
`73268338b73b4ae2edc32397ed733fb1edddf598`. It completed in 10,031.31 seconds
(2 h 47 min 11 s), with peak allocated/reserved GPU memory of 2.35/2.55 GB. Fixed-batch loss
decreased from 9.4365 to 0.01206. The final checkpoint reload passed, actions remained bounded,
and training recorded zero clipping or projection events.

| Step | Files | Bytes | Checkpoint SHA-256 |
| ---: | ---: | ---: | --- |
| 5,000 | 9 | 1,319,485,667 | `8c4f4b3…a1dcbb` |
| 10,000 | 9 | 1,319,485,668 | `12866771…6dae79` |
| 20,000 | 9 | 1,319,485,668 | `3667a125…c77e80` |

Training and offline loss are pipeline evidence only. They did not establish task competence.

## Staged stop

The Pick closed-loop gate failed both before and after the sole permitted bounded repair. The
shared, Stack, and Push full runs were therefore never started. No final split, language
intervention, VLA-JEPA, LatentGuard, SARM, PPO, or new-data work ran.

Compact source bindings are in
`artifacts/langmani_v2/phase_2c_b/training_result.json` and
`artifacts/langmani_v2/phase_2c_b/checkpoint_and_offline_diagnostics.json`.
