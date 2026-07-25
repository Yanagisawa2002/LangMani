# LangMani 2.0 Phase 2C-B SmolVLA audit

## Scope and authority

Phase 2C-B starts from
`4c38190a36ec41349ef39774ee95dc666537eed6` on
`codex/langmani-v2-phase2c-a1-bounded-act`. It consumes only
`LangManiOfficialMultiSkill-v2` with canonical package fingerprint
`sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`.
The accepted package contains 2,998 episodes and 254,200 frames. The policy
train views contain 700 Pick episodes/54,608 frames, 700 Stack
episodes/74,891 frames, and 699 Push episodes/48,219 frames.

The Phase 2C-A.1 closure remains `RESULT_B`: the ACT pipeline was valid, all
four ACT policies executed 360 closed-loop episodes and 18,000 physical
actions with zero invalid-action or simulator-error episodes, but all four
success rates were zero. ACT is closed and is not modified or retrained by
this phase.

## Official base identity

The implementation is LeRobot 0.6.0 and the pinned base is
`lerobot/smolvla_base@c83c3163b8ca9b7e67c509fffd9121e66cb96205`.

- base config SHA-256:
  `650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d`;
- base model SHA-256:
  `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`;
- nested VLM:
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`;
- nested config SHA-256:
  `ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17`;
- nested model SHA-256:
  `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`.

The audited target runtime used Python 3.12.13, PyTorch 2.11.0+cu128,
CUDA 12.8, cuDNN 9.19, Transformers 5.4, Tokenizers 0.22.2,
Safetensors 0.8, and an NVIDIA GeForce RTX 5090. Diffusers is absent and is
not imported by the pinned official SmolVLA path.

Strict loading reconstructed all 450,046,176 parameters. The loaded ratio is
100.0%; 99,880,992 parameters are trainable and 350,165,184 are frozen.
No parameter is reinitialized. The published 32-dimensional internal
state/action padding path accepts the public Panda state `[9]` and action
`[8]` feature shapes without changing a pretrained tensor shape. The only
camera feature is `observation.images.base_camera`.

## Processor and model-input boundary

The exact model input allowlist is:

- `observation.images.base_camera`;
- `observation.state`;
- `action`;
- `action_is_pad`;
- natural-language `task`.

Task IDs and every privileged simulator field are removed before
preprocessing. Immutable LeRobot camera bytes remain uint8 at the dataset
boundary and are converted exactly once to float32 `[0,1]` by division by
255 before official resizing. Visual normalization is identity, state
normalization uses accepted train-only statistics, and action normalization
is identity because the isolated bounded latent transform owns the action
representation.

All four real train views decoded their first and last boundary rows through
LeRobot/PyAV. The shared sampler processed 8,000 micro samples as
2,667 Pick, 2,667 Stack, and 2,666 Push samples, demonstrating the frozen
long-run 1/3 task balance. Natural language is the shared model's only
explicit task condition.

## Bounded physical action path

The reviewed official flow decoder is unbounded. Phase 2C-B therefore uses
the parameter-free, serialized
`smolvla_bounded_action_latent_v1` transform:

```text
physical
-> affine map to [-1,1]
-> atanh((1-epsilon) * normalized), epsilon=1e-6
-> SmolVLA latent target

predicted latent
-> tanh
-> affine map to native Panda bounds
-> physical action
```

The same transform is used in training, diagnostics, checkpoint reload, and
closed-loop inference. It performs no clipping, projection, rejection
sampling, zero replacement, or previous-action replacement. Exact native
endpoints have finite nearest-interior training targets, while infinite
synthetic predictions decode deterministically to native endpoints.

The 100,000-chunk static audit produced zero non-finite actions, lower-bound
violations, upper-bound violations, clipping events, or projection events.
Explicit `action_is_pad` tests proved that padded positions contribute zero
loss, unpadded positions contribute normally, and all-padding batches remain
finite.

## Pipeline and micro gates

The provenance-bound smoke uses training commit
`73268338b73b4ae2edc32397ed733fb1edddf598`. Its run and checkpoint manifests
bind that full commit, static-preparation fingerprint, base-audit
fingerprint, real-view fingerprint, and run-manifest fingerprint. A real
Pick batch completed forward, backward, optimizer step, checkpoint save,
strict checkpoint/processor reload, bounded nonconstant action generation,
official environment construction, and exactly one successful `env.step`.
The one-step episode timed out by design and is pipeline evidence only.

The provenance-bound 500-step Pick micro gate reduced fixed-batch loss from
9.4365 to 0.3652 and physical-action MAE from 1.0173 to 0.1369. The
500-step shared micro gate reduced fixed-batch loss from 15.0239 to 0.2604
and physical-action MAE from 1.2887 to 0.1788. Different observations
changed outputs, and changing only the shared instruction changed generated
actions by a maximum absolute physical component difference of 1.9213.
Both checkpoints reconstructed exactly and all generated actions remained
bounded. These are learning-path checks, not closed-loop competence claims.

## Target infrastructure note

The migrated container exposed both a broken GLX Vulkan ICD and a working
EGL Vulkan ICD. CUDA training was unaffected. Simulator construction is
bound at process launch to the existing
`/etc/vulkan/icd.d/my_nvidia_icd.json`; `vulkaninfo` identifies the RTX 5090,
Vulkan 1.4.329, and driver 595.71.05. The rejected default-ICD launch
performed no evaluation episode and is retained only as infrastructure
diagnosis.

No VLA-JEPA, LatentGuard, SARM, PPO, Diffusion Policy, dataset writer,
replay, ACT training, or additional model family is authorized or started.
