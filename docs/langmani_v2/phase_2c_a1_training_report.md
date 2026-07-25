# LangMani 2.0 Phase 2C-A.1 training report

## Result

The single authorized bounded-ACT repair completed all four seed-0 training
runs. Training used the frozen `LangManiOfficialMultiSkill-v2` package
fingerprint
`sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`
and implementation commit
`852658d60d8924d53fb7eb0d26cccbc4e54424ad`.

This report describes training and offline gates. Closed-loop policy quality is
reported separately and was not inferred from training losses.

## Frozen model and optimization contract

All runs used:

- seed 0;
- RGB `3x256x256` and `PandaPolicyStateV0[9]`;
- physical `pd_joint_pos float32[8]`;
- chunk size and action horizon 16;
- ResNet-18, model dimension 512, feed-forward dimension 3,200;
- four encoder, one decoder, and four VAE-encoder layers;
- eight attention heads, latent dimension 32, dropout 0.1;
- AdamW, learning rate `1e-5`, backbone learning rate `1e-5`,
  weight decay `1e-4`, and gradient-norm limit 10;
- bfloat16 and batch size 408;
- no pretrained backbone weights, augmentation, scheduler, or warmup;
- padding-masked physical-action L1 plus the unchanged ACT KL term with
  weight 10.

The three per-task models used no task condition. The shared model used only a
three-dimensional one-hot stable task ID through
`observation.environment_state`, with exactly 136 examples per task in every
full batch. This is oracle task conditioning, not language grounding.

## Bounded physical action head

`bounded_action_head_v1` maps every finite raw logit through:

```text
u = Linear(decoder_token)
z = tanh(float32(u))
alpha = 0.5 * (z + 1)
a = lerp(lower_float32, upper_float32, alpha)
```

The action-bound fingerprint is
`sha256:9838da7b5e929b8390e3b3353366ad33988529e785fae48baf6068207a7d2b06`
and the head fingerprint is
`sha256:6462a774ad6dad33d5f3853b056d16c0c63f18e882b026ea7329b708119a2fa3`.
Actions remain in native physical coordinates throughout training and
inference. No clip, projection, gripper threshold, rejection sampling, or
replacement action exists.

The independent random audit covered 100,000 chunks and 1,600,000 actions with
zero non-finite values, bound violations, clipping events, or projection
events. Its fingerprint is
`sha256:16e67f6c8c7b0889d4888c00e8d05cc448407b785014bd98c7939da5b70ba048`.
The all-padding loss was finite zero; changing padded targets by `1e9` changed
the loss by zero; unpadded rows contributed normally.

## GPU and micro-overfit gates

The real LeRobot GPU forward/backward/optimizer/save/reload/inference smoke
passed on an NVIDIA GeForce RTX 5090. The deterministic real-data Pick
micro-overfit reduced its measured error from `0.883945` to `0.137432`.
The shared micro-overfit reduced its error from `1.254733` to `0.136458`; its
fixed-observation task-condition response difference was `0.646006`.
These gates establish trainability and conditioning sensitivity only.

## Full training runs

| Model | Steps | Examples | Effective samples by task | Duration | Mean examples/s | Peak allocated / reserved |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Pick ACT | 2,680 | 1,092,160 | Pick 1,092,160 | 2,422.72 s | 2,161.51 | 1.185 / 18.835 GB |
| Stack ACT | 3,680 | 1,497,820 | Stack 1,497,820 | 3,519.72 s | 2,155.62 | 1.185 / 18.837 GB |
| Push ACT | 2,380 | 964,380 | Push 964,380 | 2,153.60 s | 2,176.44 | 1.185 / 18.841 GB |
| Shared Task-ID ACT | 8,712 | 3,554,496 | 1,184,832 for each task | 8,322.65 s | 2,144.32 | 1.183 / 18.883 GB |

The shared budget is uniform by task. Because the frozen per-task frame counts
differ, its effective passes are Pick `21.6970`, Stack `15.8208`, and Push
`24.5719`; this is the preregistered arithmetic-mean budget, not silent
reweighting.

## Checkpoint screening and selection

Four checkpoints were retained per model. All 16 completed exactly 10,000
ordered validation policy queries and passed finite-output, native-bound,
serialization, processor-identity, and degeneration filters. Validation-only
ranking selected exactly one checkpoint per model:

| Model | Run fingerprint | Selected step | Selected checkpoint fingerprint |
| --- | --- | ---: | --- |
| Pick ACT | `sha256:59ffd19955ee79d2f85508a354fb7c5148d95fccf17e9af5df27d6277b94a877` | 2,680 | `sha256:faebd0043e8dc948f283ac86f893b6f6cd520e80339705e1e868dceacf4db653` |
| Stack ACT | `sha256:25789faefc4a9940bf7c4ae3ebcae0a7814e3e85f889e27d36fcc04e8ab4ce0e` | 3,680 | `sha256:8391ba9784bc3ec3ed630d6c51bc78e0172eeef7b27bd585df0b3036178c312e` |
| Push ACT | `sha256:6583d27a51aa30660ee4402204290e99ee0220c672317692fd80b1a4511feef4` | 2,380 | `sha256:7096f41a3a2cf7d8c51f22ce8337bcf198db0b54d1b7d62ed6bba68ca2f02797` |
| Shared Task-ID ACT | `sha256:2185fd2725f40de1e28ce88656219af5fdbf9a25988b8992f2876f9b6e9e68f0` | 8,712 | `sha256:0a44b1b2f677522ae71084d8bee8c765ae472b77ac20fefc572a9cdf1dd5f355` |

The final verifier independently rehashed ten component artifacts for every
one of the 16 checkpoints. Checkpoint bytes remain external evidence and are
not committed.

## Evidence

Compact records are in
[`artifacts/langmani_v2/phase_2c_a1`](../../artifacts/langmani_v2/phase_2c_a1).
The compact artifact manifest contains 30 files and 413,952 bytes with
fingerprint
`sha256:57606d6114f2befe43be85c62ddb075bcf7d1c0eb76503b75c5c74dc0cc88d2e`.
Raw metrics and model bytes remain outside Git and are bound by content hash.
