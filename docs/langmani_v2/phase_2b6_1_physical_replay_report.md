# Phase 2B.6.1 physical replay report

## Control and target execution

The first accepted control, StackCube episode 936 Mode A repetition 1, reached environment
construction but failed before reset:

```text
RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver
```

The environment was not created, reset did not start, and zero source actions were submitted.
Consequently no canonical task outcome, contact state, lift state, placement state, terminal
continuous state, or success timing was observed for control 936.

The frozen control-failure stop rule was applied exactly:

| Run | Outcome |
| --- | --- |
| Control 936 Mode A | environment construction failed; no physical replay |
| Control 937 Mode A | not run after hard stop |
| Target 938 Mode A ×3 | not run |
| Target 938 Mode B ×2 | not run |
| Target 938 Mode C ×1 | not run |

Control 936 was not retried with a corrected environment variable. This prevents a convenient
infrastructure fix from silently consuming a second control repetition.

## Determinism and success

Episode 938 physical determinism is unverified. Fresh canonical success, stable-success timing,
and source/replay outcome agreement are all unobserved. The source itself records first success
at step 100, six trailing successful steps, and final success, but those source fields cannot
substitute for a replay.

No observation layer changed a physical result because no observation-mode replay ran. No
serialization or writer result exists. The historical Phase 2B.6 rejection remains unchanged:
its exact internal failing sub-gate was not persisted and is not reconstructed here.

## Infrastructure diagnosis

The GPU was visible as an NVIDIA GeForce RTX 5090 with driver 595.71.05. Read-only Vulkan
diagnostics showed:

| Loader selection | `vulkaninfo --summary` |
| --- | --- |
| SAPIEN process-selected packaged ICD | exit 1 |
| `/etc/vulkan/icd.d/nvidia_icd.json` (legacy GLX) | exit 1 |
| `/etc/vulkan/icd.d/my_nvidia_icd.json` (EGL library) | exit 0 |

This localizes the new failure to environment construction/runtime ICD selection. It does not
show that episode 938 is physically compatible or incompatible. The original preflight checked
GPU presence but did not construct a Vulkan instance; that gap invalidates this physical
forensic attempt and leads to Result D.
