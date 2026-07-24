# Phase 2B.6.1-R result

Phase 2B.6.1-R is `RESULT_A`: the explicit Vulkan environment contract was recovered.

This result is limited to Vulkan/SAPIEN preflight and three independent zero-step
`StackCube-v1` constructions. It does not amend Phase 2B.6 `RESULT_C`, reinterpret Phase 2B.6.1
`RESULT_D`, or establish a new trajectory, task-quality, dataset, or model result.

## Required questions

| Question | Evidence-based answer |
| --- | --- |
| 1. Which Vulkan ICDs were present? | The relevant identities were the explicit EGL-associated `/etc/vulkan/icd.d/my_nvidia_icd.json`, the system GLX-associated `/etc/vulkan/icd.d/nvidia_icd.json`, and SAPIEN's packaged automatic GLX-associated ICD; the full system inventory is recorded separately. |
| 2. Which ICD had previously worked with `vulkaninfo`? | `/etc/vulkan/icd.d/my_nvidia_icd.json`. |
| 3. Which exact ICD and environment variables were selected? | Candidate A: `VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json`, `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`, `CUDA_VISIBLE_DEVICES=0`, `VK_DRIVER_FILES`, `DISPLAY`, and `WAYLAND_DISPLAY` unset, and a fresh phase-owned `XDG_RUNTIME_DIR`. |
| 4. Did the minimal SAPIEN probe pass? | Yes, 3/3 fresh processes selected the RTX 5090 and exited cleanly. |
| 5. Did StackCube construction pass without reset or step? | Yes, 3/3 `gym.make` and close operations passed with zero explicit resets and zero steps. |
| 6. Did all three fresh processes agree? | Yes: same ICD hash, GPU, PCI identity, action contract, observation contract, and successful close. |
| 7. Was the intended RTX 5090 selected? | Yes: CUDA device 0, GPU UUID `GPU-27f8a10c-1b18-1109-85da-247ee1927024`, NVIDIA PCI `00000000:27:00.0`, SAPIEN PCI `0000:27:00.0`, driver 580.76.05. |
| 8. Did any process or GPU workload leak? | No. No phase process, renderer, writer, replay, training job, tmux session, compute process, or VRAM allocation remained. |
| 9. Was the previous default-route failure reproduced? | Yes. The unmodified Candidate C route selected the packaged GLX-associated ICD and failed `vulkaninfo` with `ERROR_INCOMPATIBLE_DRIVER`. |
| 10. Is forensic replay merely eligible to restart? | Yes. `phase2b6_1_forensic_restart_eligible=true` is the only newly opened decision state. |
| 11. Why does replay itself remain unauthorized? | This phase was explicitly zero-step and produced no evidence about controls 936/937 or target 938. A new immutable protocol and separate user authorization are still required. |
| 12. Why do production and training remain unauthorized? | No replay compatibility, accepted multi-skill dataset, training package, or model-quality evidence was produced. Eligibility for one bounded forensic restart cannot authorize Phase 2B.6 production or any policy consumer. |

## Final classification and authorization

```text
vulkan_preflight_validated=true
stackcube_zero_step_construction_validated=true
phase2b6_1_forensic_restart_eligible=true

phase2b6_1_forensic_restart_authorized=false
phase2b6_production_resume_authorized=false
accepted_multiskill_dataset_validated=false
accepted_dataset_package_created=false

act_training_eligible=false
smolvla_training_eligible=false
vla_jepa_training_eligible=false
act_training_authorized=false
smolvla_training_authorized=false
vla_jepa_training_authorized=false

production_or_forensic_replay_started=false
optimizer_created=false
backward_passes=0
optimizer_steps=0
student_policy_training_started=false
```

The result fingerprint is
`sha256:5ed07db5c9d41b7f944301ab4aef0346b2dd6bec46562ee3e92154b0e0909fa1`.
The compact 19-entry artifact manifest is
`sha256:23406ce54be296e47d0b8186fe1d5f10a714df50d800c429b43e6c778ebfba6c9`.

## Recommended next phase

If and only if separately authorized, create a new Phase 2B.6.1-v2 forensic milestone with a new
branch, output identity, and frozen protocol. It should reuse the accepted launcher and begin the
original bounded protocol again at control episode 936. This recommendation is not authorization:
this phase stops before replay, and Phase 2B.6 production and all training remain closed.
