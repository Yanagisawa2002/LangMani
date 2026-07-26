# Phase 2C-C training report

Status: complete. The sole authorized seed-0 relative-action Pick SmolVLA completed 20,000
optimizer steps and passed its training, checkpoint, and offline-diagnostic gates.

## Frozen inputs and configuration

Training consumed only the accepted `PickCube-v1` train split from
`LangManiOfficialMultiSkill-v2`, package fingerprint
`sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`.
The official base snapshot was `lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205`;
the nested VLM snapshot was
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
Only the action target and decoder changed from absolute native targets to the frozen
query-state-relative bounded residual contract.

The unchanged training settings were seed 0, AdamW, learning rate `1e-4`, batch size 4 with four
gradient-accumulation steps, bfloat16, no augmentation, frozen vision encoder, 50-action chunks,
10 inference steps, 1,000 warmup steps, and a 20,000-step schedule. The run used Python 3.12.13,
PyTorch 2.11.0+cu128, LeRobot 0.6.0, Transformers 5.4.0, and ManiSkill 3.0.1 on one NVIDIA GeForce
RTX 5090 (32,607 MiB; driver 595.71.05). Training source was commit
`65c3f11d204516f6c80e1776b8591418f08ad0fb`.

## Pre-training gates

The accepted v1 static preparation used the query state as the anchor for every valid future
target in a 50-action chunk. It replaced an invalid frame-scale diagnostic that stopped before any
optimizer step. The accepted scales were frozen at `0.6531111598014832` for the seven arm
components and `16.947166442871094` for the gripper component.

The 100,000-chunk audit covered 5,000,000 actions and 40,000,000 components with zero non-finite
values, bound violations, clipping, projection, or replacement. Maximum zero-residual identity
error was `4.76837158203125e-07`. Reconstruction covered all 70,239 accepted Pick frames and all
valid query-anchored targets in train, validation, and unseen-reset views. Maximum absolute error
was `7.152557373046875e-07`, below the frozen `1e-6` float32 tolerance, with zero violations.

The one-step real-batch smoke completed forward, backward, one optimizer step, checkpoint
save/reload, bounded nonconstant generation, and one real environment step. The separately
required 500-step micro-overfit reduced fixed-batch loss from `0.6188072` to `0.1718558` and
reconstructed physical/residual MAE from `0.5577114` to `0.0721126`; its checkpoint also passed
exact reload and a real environment step.

## Full run and checkpoint selection

The full run completed in `10,118.759` seconds (2 h 48 min 39 s). Fixed-batch loss fell from
`0.6188072` to `0.0112416`; first-action physical MAE fell from `0.5494596` to `0.0103602`.
Generated actions remained structurally bounded and nonconstant after reload.

Validation-only offline diagnostics were:

| Step | Flow loss | Residual/physical MAE | First-action MAE | Arm MAE | Gripper MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.0367567 | 0.0753111 | 0.0495288 | 0.0769772 | 0.0636485 |
| 10,000 | 0.0302738 | 0.0608117 | 0.0300023 | 0.0630968 | 0.0448162 |
| 20,000 | 0.0257504 | 0.0525535 | 0.0192859 | 0.0544713 | 0.0391289 |

The pre-registered validation-only ranking selected step 20,000. Checkpoint identities were:

- step 5,000:
  `sha256:b755ec315a160e2a3c1b920322f6573559d2091bdb7e6afab4cd6d5f36c6175e`;
- step 10,000:
  `sha256:19646f7d98e53301e45ec474543d7660e176677b0bd96303bac57d8cd0bb1a4b`;
- step 20,000:
  `sha256:f42b6cbd7ce798aef969e7c328e1ccfc7dc677fa407498c9b3b0344bc3e621f3`.

No Shared, Stack, Push, VLA-JEPA, ACT, second-seed, repair, or other architecture training ran.
