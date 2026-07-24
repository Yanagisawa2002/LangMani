# Phase 2C-A LeRobot ACT audit

The isolated target environment used Python 3.12.13, LeRobot 0.6.0, PyTorch 2.11.0+cu128,
Torchvision 0.26.0+cu128, ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium 1.2.3, NumPy 2.2.6, and
PyAV 15.1.0 on one RTX 5090 with driver 580.76.05. SmolVLA, Diffusers, and VLA-JEPA were not
loaded.

The maintained ACT implementation remained unmodified. The audit hashed the installed
configuration, model, and processor sources. It verified:

- ResNet-18 without downloaded pretrained weights;
- `chunk_size=16`, `n_action_steps=16`, model dimension 512, eight heads, four encoder layers,
  one decoder layer, and a 32D VAE latent;
- `MEAN_STD` image/state/action processing and identity processing for the shared public task
  token;
- `action_is_pad` feeding the installed `valid_mask` and zeroing padded action-loss components;
- ACT loss equal to masked component-mean L1 plus KL weighted by 10;
- save/reload of model, preprocessor, postprocessor, optimizer, and RNG state;
- deterministic evaluation with no temporal ensemble;
- no privileged policy inputs and no action clipping or projection.

One diagnostic adapter defect was found after training: the offline diagnostic passed a mapping to
a LeRobot 0.6 postprocessor that expects a policy-action tensor. Commit `a3a09d30` reused the
already-tested tensor postprocessing path and added a regression test. This changed no model,
checkpoint, dataset, statistic, or selection metric. All 16 diagnostics then completed.

The full machine audit is
`/root/autodl-tmp/langmani-phase2c-a/preflight-c0b5106-b408/act_implementation_audit.json`.
